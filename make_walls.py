"""Generates the two walls of the maze as one closed STL each (case/wall_1.stl, case/wall_2.stl).

Each wall is a slab with rectangular windows cut out, so it is a single surface
(no hidden faces between boxes, much cleaner in the visualization). It needs
    .venv/bin/pip install trimesh manifold3d
and run_optimization.py calls it on its own if the STL files do not exist yet.
Edit the numbers below to change the maze and delete the STL files (or run
this script) to regenerate them.
"""
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent / "case"

# lengths in meters
WALL_Y = (-3.2, 5.0)
WALL_Z = (-3.2, 4.6)
WALL_1_X = (2.2, 2.4)
WALL_2_X = (4.6, 4.8)
WINDOWS_1 = [                         # (y range, z range)
    ((-1.1, -0.2), (0.2, 1.1)),       # upper, left of the inlet
    ((0.4, 1.3), (-0.9, -0.1)),       # lower, to the right
    ((1.8, 2.7), (0.8, 1.8)),         # high and far to the right
]
WINDOWS_2 = [
    ((-0.3, 0.6), (-0.4, 0.5)),       # low and near the start
    ((1.0, 1.9), (0.9, 1.8)),         # in the middle, high
    ((2.3, 3.0), (-0.5, 0.3)),        # far to the right, low
]
WALLS = [("wall_1.stl", WALL_1_X, WINDOWS_1), ("wall_2.stl", WALL_2_X, WINDOWS_2)]


def write_walls():
    import trimesh      # only needed to cut the windows, not to run the optimization
    for name, x, windows in WALLS:
        slab = trimesh.creation.box(bounds=[[x[0], WALL_Y[0], WALL_Z[0]], [x[1], WALL_Y[1], WALL_Z[1]]])
        # the cutters stick out of the slab on both sides so the cut is clean
        cutters = [trimesh.creation.box(bounds=[[x[0] - 0.1, y[0], z[0]], [x[1] + 0.1, y[1], z[1]]])
                   for y, z in windows]
        wall = trimesh.boolean.difference([slab, *cutters], engine="manifold")
        assert wall.is_watertight, f"{name} is not closed"
        wall.export(CASE_DIR / name)
        print(f"{CASE_DIR / name}: {len(wall.faces)} triangles")


if __name__ == "__main__":
    write_walls()
