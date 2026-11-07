"""Animation of an optimization: every simulated design next to its pressure drop.

Run it with ParaView's batch interpreter (it needs ffmpeg to make the video):
    pvbatch animate_results.py RESULTS_DIR [--output FILE.mp4]
                               [--single ITERATION|best] [--max-iterations N]
                               [--seconds-per-iteration 0.3] [--final-hold 4]
                               [--rho 1.2] [--pressure-range MIN MAX]

The tube of each iteration is colored with the pressure on its walls (read from
the OpenFOAM case), on a color scale that is the same for every iteration (the
range of all the iterations, or --pressure-range). The obstacles are grey. The
pressure drop of every iteration and the best value so far are in a chart
floating over the upper right corner of the scene. The video ends on the best
solution alone.

With --single only one image is rendered (no ffmpeg needed), to see how a design
looks: the chart has the history up to that iteration. Default output:
RESULTS_DIR/solution_<iteration>.png. The color scale is still the one of all the
iterations, so the image matches the video.
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paraview.simple import (AssignViewToLayout, Calculator, CreateLayout, CreateView, Delete,
                             ExtractSurface, GetColorTransferFunction, GetScalarBar, MergeBlocks,
                             OpenFOAMReader, SaveScreenshot, Show, STLReader, Text,
                             TrivialProducer)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkTable

WIDTH, HEIGHT = 1920, 1080
# Direction from the center of the scene to the camera (x, y, z), z is up
CAMERA_DIRECTION = [0.5, -0.6, 0.8]
# Distance to the scene in diagonals (smaller = bigger scene) and how far the scene moves
# to the left, as a fraction of the diagonal
CAMERA_DISTANCE, SCENE_SHIFT = 1.4, 0.15
ROTATION_SPEED = 12.0   # degrees per second, the camera turns around the vertical axis of the scene
BLUE, GREEN, GREY = [0.20, 0.45, 0.85], [0.15, 0.65, 0.35], [0.55, 0.55, 0.55]
# Chart floating over the upper right corner of the scene
CHART_SIZE, CHART_MARGIN = (760, 440), 24
AXIS_TITLE_SIZE = 44    # strips left and below the chart for the axis titles, drawn by hand
TUBE_PATCHES = ["patch/inlet", "patch/outlet", "patch/wall"]


def read_history(results_dir):
    """Valid iterations only: (iteration numbers, pressure drops)."""
    with open(results_dir / "history.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    return (np.array([int(r["eval"]) for r in rows]),
            np.array([float(r["pressure_drop"]) for r in rows]))


def tube_file(results_dir, iteration):
    return str(results_dir / f"eval_{iteration:04d}" / "constant" / "triSurface" / "tube.stl")


def default_rho():
    """Density of case/config.toml, the one used to convert the pressure to Pa."""
    config = Path(__file__).resolve().parent / "case" / "config.toml"
    with open(config, "rb") as f:
        return tomllib.load(f)["fluid"]["rho"]


def tube_pressure(results_dir, iteration, rho):
    """Pipeline with the tube surface of an iteration and its wall pressure in Pa.

    Returns (reader, surface, time), the surface has the cell array "Pressure".
    """
    reader = OpenFOAMReader(FileName=str(results_dir / f"eval_{iteration:04d}" / "case.foam"))
    reader.MeshRegions = TUBE_PATCHES
    reader.CellArrays = ["p"]
    reader.UpdatePipelineInformation()
    time = max(reader.TimestepValues)        # last time written = converged solution
    surface = ExtractSurface(Input=MergeBlocks(Input=reader))
    # OpenFOAM works with kinematic pressure p / rho
    surface = Calculator(Input=surface, AttributeType="Cell Data",
                         ResultArrayName="Pressure", Function=f"p*{rho}")
    surface.UpdatePipeline(time)
    return reader, surface, time


def make_table(columns):
    table = vtkTable()
    for name, values in columns.items():
        array = numpy_to_vtk(np.asarray(values, dtype=float), deep=True)
        array.SetName(name)
        table.AddColumn(array)
    return table


def style_surface(display, color, opacity=1.0):
    """Plain color (obstacles)."""
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Opacity = opacity


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="default: RESULTS_DIR/animation.mp4")
    parser.add_argument("--single", metavar="ITERATION", default=None,
                        help="render only this iteration (a number, or 'best') as a PNG image")
    parser.add_argument("--max-iterations", type=int, default=None, metavar="N",
                        help="render only the first N valid iterations, for a quick preview")
    parser.add_argument("--seconds-per-iteration", type=float, default=0.3)
    parser.add_argument("--final-hold", type=float, default=4.0, help="seconds on the best solution")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--rotation-speed", type=float, default=ROTATION_SPEED,
                        help="degrees per second the camera turns around the scene, independent of the "
                             "iterations (0 = fixed camera)")
    parser.add_argument("--rho", type=float, default=None, help="kg/m^3, default: case/config.toml")
    parser.add_argument("--pressure-range", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="fixed color scale in Pa, default: range of all the iterations")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    iters, dps = read_history(results_dir)
    if args.max_iterations is not None:
        iters, dps = iters[:args.max_iterations], dps[:args.max_iterations]
    if len(iters) == 0:
        raise SystemExit("no valid iterations in history.csv")
    single = None          # index in the history of the only iteration to render
    if args.single is not None:
        if args.single == "best":
            single = int(np.argmin(dps))
        elif args.single.isdigit() and int(args.single) in iters:
            single = int(np.where(iters == int(args.single))[0][0])
        else:
            raise SystemExit(f"iteration '{args.single}' is not a valid iteration of history.csv")
    default_output = (f"solution_{iters[single]:04d}.png" if single is not None else "animation.mp4")
    output = (args.output or results_dir / default_output).resolve()
    rho = args.rho if args.rho is not None else default_rho()
    best_so_far = np.minimum.accumulate(dps)
    best_iter_so_far = np.array([iters[np.argmin(dps[:k + 1])] for k in range(len(dps))])

    # The scene takes the whole image, the chart is rendered apart and pasted over it
    render = CreateView("RenderView")
    render.UseColorPaletteForBackground = 0
    render.Background = [1, 1, 1]
    layout = CreateLayout("scene")
    # Same size as the final image, otherwise the fonts get scaled up when saving
    layout.SetSize(WIDTH, HEIGHT)
    AssignViewToLayout(view=render, layout=layout, hint=0)
    chart = CreateView("XYChartView")
    chart_layout = CreateLayout("chart")
    chart_layout.SetSize(CHART_SIZE[0] - AXIS_TITLE_SIZE, CHART_SIZE[1] - AXIS_TITLE_SIZE)
    AssignViewToLayout(view=chart, layout=chart_layout, hint=0)

    # Obstacles
    obstacles_stl = results_dir / "obstacles.stl"
    if obstacles_stl.exists():
        style_surface(Show(STLReader(FileNames=[str(obstacles_stl)]), render), GREY, 0.35)

    # Fixed camera and fixed color scale, looking at everything that will ever be shown
    bounds = np.array([np.inf, -np.inf] * 3)   # xmin, xmax, ymin, ymax, zmin, zmax
    pressure_min, pressure_max = np.inf, -np.inf
    for it in iters:
        reader, surface, time = tube_pressure(results_dir, it, rho)
        b = np.array(surface.GetDataInformation().GetBounds())
        bounds[[0, 2, 4]] = np.minimum(bounds[[0, 2, 4]], b[[0, 2, 4]])
        bounds[[1, 3, 5]] = np.maximum(bounds[[1, 3, 5]], b[[1, 3, 5]])
        low, high = surface.CellData["Pressure"].GetRange()
        pressure_min, pressure_max = min(pressure_min, low), max(pressure_max, high)
        Delete(surface)
        Delete(reader)
    if args.pressure_range:
        pressure_min, pressure_max = args.pressure_range
    print(f"Color scale: {pressure_min:.2f} to {pressure_max:.2f} Pa", flush=True)
    if obstacles_stl.exists():
        reader = STLReader(FileNames=[str(obstacles_stl)])
        reader.UpdatePipeline()
        b = np.array(reader.GetDataInformation().GetBounds())
        bounds[[0, 2, 4]] = np.minimum(bounds[[0, 2, 4]], b[[0, 2, 4]])
        bounds[[1, 3, 5]] = np.maximum(bounds[[1, 3, 5]], b[[1, 3, 5]])
    center = (bounds[0::2] + bounds[1::2]) / 2
    diagonal = np.linalg.norm(bounds[1::2] - bounds[0::2])
    direction0 = np.array(CAMERA_DIRECTION) / np.linalg.norm(CAMERA_DIRECTION)

    def set_camera(angle_deg):
        """Camera turned angle_deg around the vertical axis through the center of the scene."""
        a = np.radians(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        direction = rot @ direction0
        # Looking at a point to the right of the scene's center puts the scene on the left of the image
        right = np.cross(-direction, [0, 0, 1])
        focus = center + right / np.linalg.norm(right) * diagonal * SCENE_SHIFT
        render.CameraFocalPoint = focus.tolist()
        render.CameraPosition = (focus + direction * diagonal * CAMERA_DISTANCE).tolist()
        render.CameraViewUp = [0, 0, 1]

    set_camera(0)
    render.OrientationAxesVisibility = 1

    # One color map for every iteration, it never rescales itself
    lut = GetColorTransferFunction("Pressure")
    lut.ApplyPreset("Turbo", True)
    lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(pressure_min, pressure_max)

    label = Text(Text="")
    label_display = Show(label, render)
    label_display.WindowLocation = "Upper Left Corner"
    label_display.FontSize = 28
    label_display.Color = [0, 0, 0]

    # Chart with fixed axes so nothing jumps between frames
    pad = 0.12 * (dps.max() - dps.min() + 1e-9)
    chart.ChartTitle = "Pressure drop per iteration"
    chart.BottomAxisTitle = " "
    # ParaView puts the axis titles on top of the tick labels and cuts them, they are drawn by hand
    chart.LeftAxisTitle = " "
    chart.BottomAxisUseCustomRange = 1
    chart.BottomAxisRangeMinimum, chart.BottomAxisRangeMaximum = 0, int(iters.max()) + 1
    chart.LeftAxisUseCustomRange = 1
    # Room on top for the legend
    chart.LeftAxisRangeMinimum, chart.LeftAxisRangeMaximum = dps.min() - pad, dps.max() + 2.5 * pad
    chart.ShowLegend = 1
    chart.ChartTitleFontSize = 20
    chart.LeftAxisLabelFontSize = chart.BottomAxisLabelFontSize = 16
    chart.LegendFontSize = 16

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
    tube = None          # (reader, surface) shown now

    def show_tube(it):
        nonlocal tube
        if tube is not None:
            Delete(tube[1])
            Delete(tube[0])
        reader, surface, time = tube_pressure(results_dir, it, rho)
        display = Show(surface, render)
        display.Representation = "Surface"
        display.ColorArrayName = ["CELLS", "Pressure"]
        display.LookupTable = lut
        bar = GetScalarBar(lut, render)
        bar.Title, bar.ComponentTitle = "Wall pressure [Pa]", ""
        bar.TitleColor = bar.LabelColor = [0, 0, 0]
        bar.TitleFontSize, bar.LabelFontSize = 22, 20
        bar.AutomaticLabelFormat, bar.LabelFormat, bar.RangeLabelFormat = 0, "%.0f", "%.0f"
        bar.WindowLocation = "Lower Right Corner"
        bar.ScalarBarLength = 0.4
        display.SetScalarBarVisibility(render, True)
        render.ViewTime = time
        tube = (reader, surface)

    def render_chart():
        """Image of the chart with its axis titles, as it floats over the scene."""
        chart_file = frames / "chart.png"
        SaveScreenshot(str(chart_file), chart_layout, ImageResolution=[CHART_SIZE[0] - AXIS_TITLE_SIZE,
                                       CHART_SIZE[1] - AXIS_TITLE_SIZE],
                       FontScaling="Do not scale fonts")
        plot = Image.open(chart_file).convert("RGB")
        chart_image = Image.new("RGB", CHART_SIZE, "white")
        chart_image.paste(plot, (AXIS_TITLE_SIZE, 0))
        font = ImageFont.load_default(18)
        draw = ImageDraw.Draw(chart_image)
        draw.text((AXIS_TITLE_SIZE + plot.width / 2, plot.height + AXIS_TITLE_SIZE / 2),
                  "Iteration [-]", fill="black", font=font, anchor="mm")
        title = Image.new("RGB", (plot.height, AXIS_TITLE_SIZE), "white")
        ImageDraw.Draw(title).text((plot.height / 2, AXIS_TITLE_SIZE / 2), "Pressure drop [Pa]",
                                   fill="black", font=font, anchor="mm")
        chart_image.paste(title.rotate(90, expand=True), (0, 0))
        chart_file.unlink()
        return chart_image

    def snapshot(n, chart_image):
        scene_file = frames / f"scene_{n:05d}.png"
        SaveScreenshot(str(scene_file), layout, ImageResolution=[WIDTH, HEIGHT],
                       FontScaling="Do not scale fonts")
        image = Image.open(scene_file).convert("RGB")
        # Slightly see-through, so it floats over the scene instead of cutting it
        image.paste(chart_image, (WIDTH - CHART_SIZE[0] - CHART_MARGIN, CHART_MARGIN),
                    Image.new("L", CHART_SIZE, 215))
        image.save(frames / f"frame_{n:05d}.png")
        scene_file.unlink()

    if single is not None:
        k = single
        show_tube(iters[k])
        label.Text = (f"Iteration {iters[k]}:  {dps[k]:.2f} Pa\n"
                      f"Best so far:  {best_so_far[k]:.2f} Pa (iteration {best_iter_so_far[k]})")
        draw_chart(k)
        snapshot(0, render_chart())
        shutil.copyfile(frames / "frame_00000.png", output)
        shutil.rmtree(frames)
        print(f"Image written to {output}")
        return

    # One video frame per 1/fps seconds. The iteration shown depends on the time, and the camera
    # angle too, but independently: the camera keeps turning whatever the iteration.
    n_iter_frames = round(len(iters) * args.seconds_per_iteration * args.fps)
    n_total = n_iter_frames + round(args.final_hold * args.fps)
    state, chart_image = None, None
    for n in range(n_total):
        t = n / args.fps
        k = int(t / args.seconds_per_iteration) if n < n_iter_frames else None   # None = final image
        if k is not None:
            k = min(k, len(iters) - 1)
        if k != state:
            state = k
            if producer is not None:
                Delete(producer)
            if k is not None:
                show_tube(iters[k])
                label.Text = (f"Iteration {iters[k]}:  {dps[k]:.2f} Pa\n"
                              f"Best so far:  {best_so_far[k]:.2f} Pa (iteration {best_iter_so_far[k]})")
                producer = draw_chart(k)
                print(f"iteration {k + 1}/{len(iters)}", flush=True)
            else:
                # Final image: only the best design and the whole history
                k_best = int(np.argmin(dps))
                show_tube(iters[k_best])
                label.Text = f"Best solution:  {dps[k_best]:.2f} Pa (iteration {iters[k_best]})"
                producer = draw_chart(len(iters) - 1)
                print("final image", flush=True)
            chart_image = render_chart()
        set_camera(args.rotation_speed * t)
        snapshot(n, chart_image)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", str(frames / "frame_%05d.png"),
                    "-pix_fmt", "yuv420p", str(output)], check=True)
    shutil.rmtree(frames)
    print(f"Animation written to {output}")


main()
