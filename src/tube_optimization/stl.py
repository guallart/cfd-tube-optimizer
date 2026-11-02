import numpy as np


def write_stl(path, solids):
    """Writes an ASCII STL with one named solid per entry.

    solids is a dict {name: triangles}, with triangles of shape (n, 3, 3).
    This is the format OpenFOAM reads to get the named patches.
    """
    with open(path, "w") as f:
        for name, tris in solids.items():
            f.write(f"solid {name}\n")
            for tri in tris:
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                n = n / np.linalg.norm(n)
                f.write(f"  facet normal {n[0]:e} {n[1]:e} {n[2]:e}\n")
                f.write("    outer loop\n")
                for v in tri:
                    f.write(f"      vertex {v[0]:e} {v[1]:e} {v[2]:e}\n")
                f.write("    endloop\n  endfacet\n")
            f.write(f"endsolid {name}\n")
