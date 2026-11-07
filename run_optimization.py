"""Optimizes a tube between two fixed ports, avoiding a few obstacles.

Edit the section below to define another case and run it with:
    .venv/bin/python run_optimization.py [--max-evaluations N]
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

from tube_optimization import Box, MeshObstacle, Sphere, TubeOptimizer  # noqa: E402
from make_walls import (CASE_DIR, WALL_1_X, WALL_2_X, WALL_Z, WALLS, WINDOWS_1,  # noqa: E402
                        WINDOWS_2, write_walls)

# lengths in meters
INLET_POINT =      [0.0, 0.0, 0.0]
INLET_DIRECTION =  [1.0, 0.0, 0.0]     # flows in along +x
OUTLET_POINT =     [7.2, 2.4, 1.6]
OUTLET_DIRECTION = [0.0, 1.0, 0.0]     # and leaves along +y, so the tube has to turn
TUBE_RADIUS = 0.1
BEZIER_DEGREE = 7                     # at least 3, every extra degree adds a free control point

if not all((CASE_DIR / name).exists() for name, _, _ in WALLS):
    write_walls()
WALL_OBSTACLES = [MeshObstacle(CASE_DIR / name, cell_size=0.025) for name, _, _ in WALLS]

OBSTACLES = [
    # Before wall 1: spheres that hide the straight way and half close two windows
    Sphere(center=[1.0, 0.0, 0.1], radius=0.35),
    Sphere(center=[1.6, -0.8, 0.8], radius=0.25),
    Sphere(center=[1.7, 1.1, -0.6], radius=0.25),
    Box(corner_min=[0.8, 1.7, 0.3], corner_max=[1.2, 2.2, 1.9]),
    # Inside the windows of wall 1: a bar and a sphere that leave a narrower opening
    Box(corner_min=[WALL_1_X[0], 1.8, 1.25], corner_max=[WALL_1_X[1], 2.7, 1.45]),
    Sphere(center=[2.3, 0.85, -0.5], radius=0.22),
    # Corridor between the walls: columns, a floating block and spheres
    Box(corner_min=[3.0, 0.2, WALL_Z[0]], corner_max=[3.4, 0.7, 1.2]),    # column from the floor
    Box(corner_min=[3.5, 1.8, 0.4], corner_max=[3.9, 2.3, WALL_Z[1]]),    # column from the ceiling
    Box(corner_min=[3.4, -0.7, 0.9], corner_max=[3.9, 0.0, 1.4]),         # floating block
    Sphere(center=[3.3, 1.3, 0.0], radius=0.3),
    Sphere(center=[3.0, 2.8, 0.9], radius=0.35),
    Sphere(center=[3.6, -1.1, -0.3], radius=0.3),
    # After wall 2: the outlet is at the back, with a few things in front
    Box(corner_min=[5.6, -0.5, WALL_Z[0]], corner_max=[6.0, 1.0, 0.9]),
    Sphere(center=[6.2, 2.2, 0.5], radius=0.3),
    Sphere(center=[5.8, 3.0, 1.8], radius=0.3),
    Sphere(center=[5.4, 1.4, 1.8], radius=0.25),
    *WALL_OBSTACLES,
    # Any closed surface modelled in other software can be added as well. Here a
    # model exported in millimeters, moved 1 m along x:
    # MeshObstacle("case/my_obstacle.stl", scale=0.001, translate=[1.0, 0.0, 0.0]),
]

def window_point(wall_x, window):
    """The point of a window (in the middle of the wall) with the most room around it."""
    (y0, y1), (z0, z1) = window
    margin = TUBE_RADIUS * 1.5
    Y, Z = np.meshgrid(np.linspace(y0 + margin, y1 - margin, 25),
                       np.linspace(z0 + margin, z1 - margin, 25), indexing="ij")
    points = np.column_stack([np.full(Y.size, sum(wall_x) / 2), Y.ravel(), Z.ravel()])
    room = np.min([o.signed_distance(points) for o in OBSTACLES], axis=0)
    return points[np.argmax(room)]


# The starting tube goes through one window of each wall: the 9 combinations are tried
# (geometry only, no simulations) and the shortest valid one is the starting design.
# Starting from the way around the walls, the optimizer would never find the windows.
START_ROUTES = [[window_point(WALL_1_X, w1), window_point(WALL_2_X, w2)]
                for w1, w2 in itertools.product(WINDOWS_1, WINDOWS_2)]

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
        results_dir=results_dir, start_routes=START_ROUTES)
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
