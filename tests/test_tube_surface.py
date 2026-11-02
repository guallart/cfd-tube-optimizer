import numpy as np

from tube_optimization import BezierCurve, TubeSurface


def make_tube():
    curve = BezierCurve([[0, 0, 0], [1, 2, 0], [3, -1, 1], [4, 0, 0]])
    return TubeSurface(curve, radius=0.1, n_axial=60, n_circ=24)


def test_bezier_endpoints_and_degree():
    curve = BezierCurve([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
    assert curve.degree == 2
    assert np.allclose(curve.evaluate([0, 1]), [[0, 0, 0], [2, 0, 0]])
    assert np.allclose(curve.tangent(0.0), [[1 / np.sqrt(2), 1 / np.sqrt(2), 0]])


def test_surface_is_closed_and_outward():
    patches = make_tube().build()
    assert set(patches) == {"inlet", "outlet", "tube"}
    tris = np.concatenate(list(patches.values()))
    # Closed surface: every edge is shared by exactly two triangles
    edges = {}
    for tri in tris:
        for i in range(3):
            e = tuple(sorted((tuple(np.round(tri[i], 9)), tuple(np.round(tri[(i + 1) % 3], 9)))))
            edges[e] = edges.get(e, 0) + 1
    assert set(edges.values()) == {2}
    # Outward normals: the signed volume must be positive
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    assert vol > 0


def test_write_stl(tmp_path):
    path = tmp_path / "tube.stl"
    make_tube().write_stl(path)
    text = path.read_text()
    for name in ("inlet", "outlet", "tube"):
        assert f"solid {name}" in text and f"endsolid {name}" in text
