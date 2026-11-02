import numpy as np
import pytest

from tube_optimization import Box, MeshObstacle, Sphere, read_stl, write_obstacles_stl
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


@pytest.mark.parametrize("solver", ["cobyla", "cma-es"])
def test_optimizer_avoids_obstacle(tmp_path, solver):
    # The straight path between the ports goes right through the sphere
    obstacles = [Sphere([1.5, 1.0, 0.25], 0.6)]

    class Strict(FakeOptimizer):
        simulated = 0

        def _simulate(self, tube, run_dir):
            # Geometry is a constraint: an invalid design never reaches the CFD
            centerline = tube.curve.evaluate(np.linspace(0, 1, 300))
            assert obstacles[0].signed_distance(centerline).min() >= 0.05
            Strict.simulated += 1
            return super()._simulate(tube, run_dir)

    opt = Strict(
        inlet_point=[0, 0, 0], inlet_direction=[1, 0, 0],
        outlet_point=[3, 2, 0.5], outlet_direction=[1, 0, 0],
        radius=0.05, degree=5, obstacles=obstacles, results_dir=tmp_path / "opt",
        config={"optimizer": {"solver": solver}})
    result = opt.run(max_evaluations=150)
    centerline = result.curve.evaluate(np.linspace(0, 1, 300))
    assert obstacles[0].signed_distance(centerline).min() >= 0.05
    assert Strict.simulated == result.n_evaluations
    assert (tmp_path / "opt" / "obstacles.stl").exists()


# --- obstacles from a surface mesh (STL)

def test_read_stl_ascii_and_binary(tmp_path):
    tris = Box([0, 0, 0], [1, 2, 3]).triangles()
    write_obstacles_stl([Box([0, 0, 0], [1, 2, 3])], tmp_path / "ascii.stl")
    assert np.allclose(read_stl(tmp_path / "ascii.stl"), tris)

    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tris), dtype=record)
    data["vertices"] = tris
    (tmp_path / "binary.stl").write_bytes(bytes(80) + len(tris).to_bytes(4, "little") + data.tobytes())
    assert np.allclose(read_stl(tmp_path / "binary.stl"), tris)


@pytest.mark.parametrize("analytic", [Sphere([1.0, 0.5, -0.5], 0.8), Box([0, 0, 0], [1.5, 1.0, 0.6])])
def test_mesh_obstacle_matches_the_analytic_distance(analytic):
    mesh = MeshObstacle(analytic.triangles(), cell_size=0.02)
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 2.5, size=(3000, 3))
    exact, approx = analytic.signed_distance(points), mesh.signed_distance(points)
    # Accurate where a tube could be touching it (the sphere mesh is a polyhedron,
    # so it is slightly inside the true sphere). Far away it only has to be large.
    near = exact < 0.15
    assert near.sum() > 50
    assert np.abs(exact - approx)[near].max() < 0.03
    assert np.all(approx[~near] > 0.1)
    assert np.all(approx[exact < -0.05] < 0) and np.all(approx[exact > 0.05] > 0)


def test_mesh_obstacle_from_file_with_scale_and_translate(tmp_path):
    # A 1 m box modelled in millimeters, moved 2 m along x
    write_obstacles_stl([Box([0, 0, 0], [1000, 1000, 1000])], tmp_path / "box.stl")
    mesh = MeshObstacle(tmp_path / "box.stl", scale=0.001, translate=[2, 0, 0], cell_size=0.02)
    assert mesh.signed_distance([[2.5, 0.5, 0.5]])[0] < -0.4
    assert abs(mesh.signed_distance([[4.0, 0.5, 0.5]])[0] - 1.0) < 0.03
    assert abs(mesh.signed_distance([[100.0, 0.5, 0.5]])[0] - 97.0) < 0.03   # far outside the grid


def test_mesh_obstacle_warns_if_open():
    tris = Box([0, 0, 0], [1, 1, 1]).triangles()[:-2]    # one face missing
    with pytest.warns(UserWarning, match="closed"):
        MeshObstacle(tris, cell_size=0.05)


def test_mesh_obstacle_works_in_the_optimizer(tmp_path):
    sphere = Sphere([1.5, 1.0, 0.25], 0.6)
    obstacle = MeshObstacle(sphere.triangles(), cell_size=0.02)
    opt = FakeOptimizer(
        inlet_point=[0, 0, 0], inlet_direction=[1, 0, 0],
        outlet_point=[3, 2, 0.5], outlet_direction=[1, 0, 0],
        radius=0.05, degree=5, obstacles=[obstacle], results_dir=tmp_path / "opt", config={})
    result = opt.run(max_evaluations=100)
    centerline = result.curve.evaluate(np.linspace(0, 1, 300))
    assert sphere.signed_distance(centerline).min() >= 0.05 - 0.03
    assert "solid obstacle_0" in (tmp_path / "opt" / "obstacles.stl").read_text()
