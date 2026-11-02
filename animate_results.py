"""Animation of an optimization: every simulated design next to its pressure drop.

Run it with ParaView's batch interpreter (it needs ffmpeg to make the video):
    pvbatch animate_results.py RESULTS_DIR [--output FILE.mp4]
                               [--seconds-per-iteration 0.3] [--final-hold 4]

Left: the tube of each iteration (blue, green when it is a new best), the best
design so far (transparent green) and the obstacles (grey). Right: the pressure
drop of every iteration and the best value reached so far. The video ends on
the best solution alone.
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from paraview.simple import (AssignViewToLayout, CreateLayout, CreateView, Delete,
                             SaveScreenshot, Show, STLReader, Text, TrivialProducer)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkTable

WIDTH, HEIGHT = 1920, 1080
# Direction from the center of the scene to the camera (x, y, z), z is up
CAMERA_DIRECTION = [0.5, -0.6, 0.8]
BLUE, GREEN, GREY = [0.20, 0.45, 0.85], [0.15, 0.65, 0.35], [0.55, 0.55, 0.55]


def read_history(results_dir):
    """Valid iterations only: (iteration numbers, pressure drops)."""
    with open(results_dir / "history.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    return (np.array([int(r["eval"]) for r in rows]),
            np.array([float(r["pressure_drop"]) for r in rows]))


def tube_file(results_dir, iteration):
    return str(results_dir / f"eval_{iteration:04d}" / "constant" / "triSurface" / "tube.stl")


def make_table(columns):
    table = vtkTable()
    for name, values in columns.items():
        array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
        array.SetName(name)
        table.AddColumn(array)
    return table


def style_surface(display, color, opacity=1.0):
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Opacity = opacity


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="default: RESULTS_DIR/animation.mp4")
    parser.add_argument("--seconds-per-iteration", type=float, default=0.3)
    parser.add_argument("--final-hold", type=float, default=4.0, help="seconds on the best solution")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = (args.output or results_dir / "animation.mp4").resolve()
    iters, dps = read_history(results_dir)
    if len(iters) == 0:
        raise SystemExit("no valid iterations in history.csv")
    best_so_far = np.minimum.accumulate(dps)
    best_iter_so_far = np.array([iters[np.argmin(dps[:k + 1])] for k in range(len(dps))])

    # Views: 3D on the left, chart on the right
    render = CreateView("RenderView")
    render.UseColorPaletteForBackground = 0
    render.Background = [1, 1, 1]
    chart = CreateView("XYChartView")
    layout = CreateLayout("animation")
    # Same size as the final image, otherwise the fonts get scaled up when saving
    layout.SetSize(WIDTH, HEIGHT)
    layout.SplitHorizontal(0, 0.55)
    AssignViewToLayout(view=render, layout=layout, hint=1)
    AssignViewToLayout(view=chart, layout=layout, hint=2)

    # Obstacles
    obstacles_stl = results_dir / "obstacles.stl"
    if obstacles_stl.exists():
        style_surface(Show(STLReader(FileNames=[str(obstacles_stl)]), render), GREY, 0.35)

    # Current tube and best tube so far share the reader slots, only the file changes
    current = STLReader(FileNames=[tube_file(results_dir, iters[0])])
    best = STLReader(FileNames=[tube_file(results_dir, iters[0])])
    current_display = Show(current, render)
    best_display = Show(best, render)

    # Fixed camera, looking at everything that will ever be shown
    bounds = np.array([np.inf, -np.inf] * 3)   # xmin, xmax, ymin, ymax, zmin, zmax
    for it in iters:
        current.FileNames = [tube_file(results_dir, it)]
        current.UpdatePipeline()
        b = np.array(current.GetDataInformation().GetBounds())
        bounds[[0, 2, 4]] = np.minimum(bounds[[0, 2, 4]], b[[0, 2, 4]])
        bounds[[1, 3, 5]] = np.maximum(bounds[[1, 3, 5]], b[[1, 3, 5]])
    if obstacles_stl.exists():
        reader = STLReader(FileNames=[str(obstacles_stl)])
        reader.UpdatePipeline()
        b = np.array(reader.GetDataInformation().GetBounds())
        bounds[[0, 2, 4]] = np.minimum(bounds[[0, 2, 4]], b[[0, 2, 4]])
        bounds[[1, 3, 5]] = np.maximum(bounds[[1, 3, 5]], b[[1, 3, 5]])
    center = (bounds[0::2] + bounds[1::2]) / 2
    diagonal = np.linalg.norm(bounds[1::2] - bounds[0::2])
    direction = np.array(CAMERA_DIRECTION) / np.linalg.norm(CAMERA_DIRECTION)
    render.CameraFocalPoint = center.tolist()
    # The view is almost square and the camera angle is 30 degrees, so stay at about 2 diagonals
    render.CameraPosition = (center + direction * diagonal * 2.0).tolist()
    render.CameraViewUp = [0, 0, 1]
    render.OrientationAxesVisibility = 1

    label = Text(Text="")
    label_display = Show(label, render)
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize = 14
    label_display.Color = [0, 0, 0]

    # Chart with fixed axes so nothing jumps between frames
    pad = 0.05 * (dps.max() - dps.min() + 1e-9)
    # No axis titles, they overlap the tick labels. The title says it all.
    chart.ChartTitle = "Pressure drop [Pa] per iteration"
    chart.BottomAxisTitle = " "
    chart.LeftAxisTitle = " "
    chart.BottomAxisUseCustomRange = 1
    chart.BottomAxisRangeMinimum, chart.BottomAxisRangeMaximum = 0, int(iters.max()) + 1
    chart.LeftAxisUseCustomRange = 1
    chart.LeftAxisRangeMinimum, chart.LeftAxisRangeMaximum = dps.min() - pad, dps.max() + pad
    chart.ShowLegend = 1
    chart.ChartTitleFontSize = 14
    chart.LeftAxisLabelFontSize = chart.BottomAxisLabelFontSize = 10
    chart.LegendFontSize = 11

    def draw_chart(k):
        """Chart with the first k+1 valid iterations."""
        producer = TrivialProducer()
        producer.GetClientSideObject().SetOutput(make_table({
            "Iteration": iters[:k + 1],
            "Pressure drop": dps[:k + 1],
            "Best so far": best_so_far[:k + 1]}))
        display = Show(producer, chart)
        display.UseIndexForXAxis = 0
        display.XArrayName = "Iteration"
        display.SeriesVisibility = ["Pressure drop", "Best so far"]
        # ParaView wants every value of these list properties as a string
        display.SeriesColor = ["Pressure drop", *map(str, BLUE), "Best so far", *map(str, GREEN)]
        display.SeriesLineStyle = ["Pressure drop", "0", "Best so far", "1"]
        display.SeriesLineThickness = ["Pressure drop", "1", "Best so far", "4"]
        display.SeriesMarkerStyle = ["Pressure drop", "4", "Best so far", "0"]
        display.SeriesMarkerSize = ["Pressure drop", "9", "Best so far", "1"]
        return producer

    # One image per iteration, then the final image of the best solution
    frames = Path(tempfile.mkdtemp(prefix="frames_"))
    producer = None

    def snapshot(n):
        SaveScreenshot(str(frames / f"frame_{n:05d}.png"), layout,
                       SaveAllViews=1, ImageResolution=[WIDTH, HEIGHT])

    for k, it in enumerate(iters):
        is_new_best = dps[k] <= best_so_far[k] and (k == 0 or dps[k] < best_so_far[k - 1])
        current.FileNames = [tube_file(results_dir, it)]
        best.FileNames = [tube_file(results_dir, best_iter_so_far[k])]
        style_surface(current_display, GREEN if is_new_best else BLUE)
        style_surface(best_display, GREEN, 0.3)
        best_display.Visibility = 0 if is_new_best else 1
        label.Text = (f"Iteration {it}:  {dps[k]:.2f} Pa\n"
                      f"Best so far:  {best_so_far[k]:.2f} Pa (iteration {best_iter_so_far[k]})")
        if producer is not None:
            Delete(producer)
        producer = draw_chart(k)
        snapshot(k)
        print(f"frame {k + 1}/{len(iters)}", flush=True)

    # Final image: only the best design and the whole history
    k_best = int(np.argmin(dps))
    best.FileNames = [tube_file(results_dir, iters[k_best])]
    style_surface(best_display, GREEN, 1.0)
    best_display.Visibility = 1
    current_display.Visibility = 0
    label.Text = f"Best solution:  {dps[k_best]:.2f} Pa (iteration {iters[k_best]})"
    Delete(producer)
    draw_chart(len(iters) - 1)
    n_hold = max(1, round(args.final_hold / args.seconds_per_iteration))
    snapshot(len(iters))
    for extra in range(1, n_hold):
        shutil.copyfile(frames / f"frame_{len(iters):05d}.png",
                        frames / f"frame_{len(iters) + extra:05d}.png")

    # Each image lasts seconds_per_iteration, ffmpeg repeats them up to the output fps
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", f"{1 / args.seconds_per_iteration:.6f}",
                    "-i", str(frames / "frame_%05d.png"),
                    "-r", str(args.fps), "-pix_fmt", "yuv420p", str(output)], check=True)
    shutil.rmtree(frames)
    print(f"Animation written to {output}")


main()
