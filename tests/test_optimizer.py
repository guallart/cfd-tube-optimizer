import numpy as np
import pytest

from tube_optimization import NotConvergedError, TubeOptimizer


class FakeOptimizer(TubeOptimizer):
    """Replaces the CFD by a cheap model: path length plus a bending penalty."""

    def _simulate(self, tube, run_dir):
        t = np.linspace(0, 1, 200)
        c = tube.curve
        length = np.linalg.norm(np.diff(c.evaluate(t), axis=0), axis=1).sum()
        bending = (c.curvature(t) ** 2).mean()
        return length + 0.1 * bending


def make(cls=FakeOptimizer, tmp_path=None, **kwargs):
    args = dict(inlet_point=[0, 0, 0], inlet_direction=[1, 0, 0],
                outlet_point=[3, 2, 0.5], outlet_direction=[0, 1, 0],
                radius=0.05, degree=5, results_dir=tmp_path / "opt", config={})
    args.update(kwargs)
    return cls(**args)


def test_curve_respects_ports_and_directions(tmp_path):
    opt = make(tmp_path=tmp_path)
    x = np.random.default_rng(0).normal(0, 0.2, len(opt.initial_guess()))
    x[:2] = [0.3, 0.4]
    curve = opt.build_curve(x)
    assert curve.degree == 5
    assert np.allclose(curve.evaluate([0, 1]), [[0, 0, 0], [3, 2, 0.5]])
    assert np.allclose(curve.tangent(0.0), [[1, 0, 0]])
    assert np.allclose(curve.tangent(1.0), [[0, 1, 0]])


def test_degree_three_is_the_minimum(tmp_path):
    with pytest.raises(ValueError):
        make(tmp_path=tmp_path, degree=2)
    assert len(make(tmp_path=tmp_path, degree=3).initial_guess()) == 2


def test_run_improves_the_initial_design(tmp_path):
    reference = make(tmp_path=tmp_path, results_dir=tmp_path / "reference")
    initial = reference.evaluate(reference.initial_guess())
    opt = make(tmp_path=tmp_path)
    result = opt.run(max_evaluations=150)
    assert result.pressure_drop < initial
    assert np.allclose(result.curve.evaluate([0, 1]), [[0, 0, 0], [3, 2, 0.5]])
    assert (tmp_path / "opt" / "history.csv").exists()


def test_invalid_designs_are_penalised_not_fatal(tmp_path):
    class Flaky(FakeOptimizer):
        def _simulate(self, tube, run_dir):
            # Pretend that any design bending towards negative z does not converge
            if tube.curve.control_points[:, 2].min() < -0.05:
                raise NotConvergedError("fake")
            return super()._simulate(tube, run_dir)

    result = make(Flaky, tmp_path).run(max_evaluations=80)
    assert result.curve.control_points[:, 2].min() >= -0.05
    assert "not_converged" in (tmp_path / "opt" / "history.csv").read_text()


def test_invalid_initial_design_raises(tmp_path):
    # Radius much larger than the curvature radius of the initial path
    with pytest.raises(RuntimeError):
        make(tmp_path=tmp_path, radius=5.0).run()
