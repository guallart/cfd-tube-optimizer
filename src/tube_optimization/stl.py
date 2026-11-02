import re
from pathlib import Path

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


def read_stl(path):
    """Reads an STL file, binary or ASCII, and returns its triangles, shape (n, 3, 3).

    Several solids in one file are merged. The normals in the file are ignored.
    """
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if len(data) == 84 + 50 * n:
            record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, dtype=record, count=n, offset=84)["vertices"].astype(float)

    numbers = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
    if not numbers or len(numbers) % 3:
        raise ValueError(f"{path} is not a valid STL file")
    return np.array(numbers, dtype=float).reshape(-1, 3, 3)
