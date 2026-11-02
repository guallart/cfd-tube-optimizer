"""Optimizes a tube between two fixed ports, avoiding a few obstacles.

Edit the section below to define another case and run it with:
    .venv/bin/python run_optimization.py [--max-evaluations N]
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from tube_optimization import Box, MeshObstacle, Sphere, TubeOptimizer  # noqa: E402

# lengths in meters
INLET_POINT =      [0.0, 0.0, 0.0]
INLET_DIRECTION =  [1.0, 0.0, 0.0]     # direction of the flow when it enters
OUTLET_POINT =     [4.0, 1.0, 1.0]
OUTLET_DIRECTION = [1.0, 0.0, 0.0]    # direction of the flow when it leaves
TUBE_RADIUS = 0.1
BEZIER_DEGREE = 7                     # at least 3, every extra degree adds a free control point

# A sphere right in front of the inlet and another one right in front of the
# outlet. Between them, a square frame at x = 2.0..2.2 made of 4 aligned boxes
# that leave a single, wider square opening (y and z from 0 to 1) for the tube.
WALL_X = (2.0, 2.2)
WINDOW_Y = (0.0, 1.0)
WINDOW_Z = (0.0, 1.0)
FRAME_LO, FRAME_HI = -0.4, 1.4        # outer extent of the frame in y and z
OBSTACLES = [
    Sphere(center=[0.8, 0.0, 0.0], radius=0.3),
    Sphere(center=[3.0, 1.0, 1.0], radius=0.3),
    Box(corner_min=[WALL_X[0], FRAME_LO, FRAME_LO], corner_max=[WALL_X[1], WINDOW_Y[0], FRAME_HI]),         # side
    Box(corner_min=[WALL_X[0], WINDOW_Y[1], FRAME_LO], corner_max=[WALL_X[1], FRAME_HI, FRAME_HI]),         # side
    Box(corner_min=[WALL_X[0], WINDOW_Y[0], FRAME_LO], corner_max=[WALL_X[1], WINDOW_Y[1], WINDOW_Z[0]]),   # bottom
    Box(corner_min=[WALL_X[0], WINDOW_Y[0], WINDOW_Z[1]], corner_max=[WALL_X[1], WINDOW_Y[1], FRAME_HI]),   # top
    # Any closed surface modelled in other software can be added as well. Here a
    # model exported in millimeters, moved 1 m along x:
    # MeshObstacle("case/my_obstacle.stl", scale=0.001, translate=[1.0, 0.0, 0.0]),
]

MAX_EVALUATIONS = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-evaluations", type=int, default=MAX_EVALUATIONS)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="default: results/optimization_<date>_<time>")
    args = parser.parse_args()

    results_dir = args.results_dir or (
        PROJECT_DIR / "results" / f"optimization_{datetime.now():%Y%m%d_%H%M%S}")

    optimizer = TubeOptimizer(
        INLET_POINT, INLET_DIRECTION, OUTLET_POINT, OUTLET_DIRECTION,
        radius=TUBE_RADIUS, degree=BEZIER_DEGREE, obstacles=OBSTACLES,
        results_dir=results_dir)
    result = optimizer.run(max_evaluations=args.max_evaluations)

    # Keep the best design in a readable file next to the cases
    best = {
        "pressure_drop_Pa": result.pressure_drop,
        "control_points": result.curve.control_points.tolist(),
        "radius": TUBE_RADIUS,
        "case": result.run_dir.name,
        "n_evaluations": result.n_evaluations,
    }
    (results_dir / "best.json").write_text(json.dumps(best, indent=2))

    print(f"\nBest pressure drop: {result.pressure_drop:.4f} Pa "
          f"after {result.n_evaluations} designs")
    print(f"Best case:  {result.run_dir}/case.foam")
    print(f"Obstacles:  {results_dir}/obstacles.stl")
    print(f"History:    {results_dir}/history.csv")


if __name__ == "__main__":
    main()
