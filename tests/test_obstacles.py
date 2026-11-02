import numpy as np
import pytest

from tube_optimization import Box, Sphere, write_obstacles_stl
from test_optimizer import FakeOptimizer


def test_sphere_signed_distance():
    s = Sphere([1, 0, 0], 2.0)
    assert np.allclose(s.signed_distance([[1, 0, 0], [3, 0, 0], [6, 0, 0]]), [-2, 0, 3])


def test_box_signed_distance():
    b = Box([0, 0, 0], [2, 2, 2])
    # center, face, and a point diagonal to a corner (distance sqrt(3))
    assert np.allclose(b.signed_distance([[1, 1, 1], [3, 1, 1], [3, 3, 3]]),
                       [-1, 1, np.sqrt(3)])


@pytest.mark.parametrize("obstacle", [Sphere([1, 2, 3], 1.5), Box([0, 0, 0], [1, 2, 3])])
def test_surface_is_closed_and_outward(obstacle):
    tris = obstacle.triangles()
    edges = {}
    for tri in tris:
        for i in range(3):
            e = tuple(sorted((tuple(np.round(tri[i], 9)), tuple(np.round(tri[(i + 1) % 3], 9)))))
            edges[e] = edges.get(e, 0) + 1
    assert set(edges.values()) == {2}
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    assert vol > 0


def test_write_obstacles_stl(tmp_path):
    write_obstacles_stl([Sphere([0, 0, 0], 1), Box([2, 2, 2], [3, 3, 3])], tmp_path / "o.stl")
    text = (tmp_path / "o.stl").read_text()
    assert "solid obstacle_0" in text and "endsolid obstacle_1" in text
    assert "nan" not in text


def test_optimizer_avoids_obstacle(tmp_path):
    # The straight path between the ports goes right through the sphere
    obstacles = [Sphere([1.5, 1.0, 0.25], 0.6)]
    opt = FakeOptimizer(
        inlet_point=[0, 0, 0], inlet_direction=[1, 0, 0],
        outlet_point=[3, 2, 0.5], outlet_direction=[1, 0, 0],
        radius=0.05, degree=5, obstacles=obstacles, results_dir=tmp_path / "opt", config={})
    result = opt.run(max_evaluations=150)
    centerline = result.curve.evaluate(np.linspace(0, 1, 300))
    assert obstacles[0].signed_distance(centerline).min() >= 0.05
    assert (tmp_path / "opt" / "obstacles.stl").exists()
    assert "obstacle" in (tmp_path / "opt" / "history.csv").read_text()
