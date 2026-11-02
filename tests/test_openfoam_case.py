import pytest

from tube_optimization import BezierCurve, OpenFOAMCase, NotConvergedError, TubeSurface, has_converged, read_pressure_drop


def test_create_case(tmp_path):
    curve = BezierCurve([[0, 0, 0], [1, 0.5, 0], [2, -0.5, 0.3], [3, 0, 0]])
    case = OpenFOAMCase(TubeSurface(curve, radius=0.1), tmp_path / "run")
    case.create()
    run = tmp_path / "run"
    assert (run / "constant" / "triSurface" / "tube.stl").exists()
    assert (run / "Allrun").stat().st_mode & 0o100
    # No placeholder should survive in any file
    for f in run.rglob("*"):
        if f.is_file() and f.suffix != ".stl":
            assert "{{" not in f.read_text(), f


def _fake_patch_average(run_dir, patch, times_and_values):
    # Mimics the layout and format written by the patchAverage function object
    for start, rows in times_and_values.items():
        d = run_dir / "postProcessing" / f"patchAverage(patch={patch},fields=(pU))" / str(start)
        d.mkdir(parents=True)
        lines = ["# Area average", "# Time p U"]
        lines += [f"{t}\t{p:.8e}\t(1.0 0.0 0.0)" for t, p in rows]
        (d / "surfaceFieldValue.dat").write_text("\n".join(lines) + "\n")


def _fake_log(run_dir, converged):
    end = "SIMPLE solution converged in 131 iterations\n\nEnd\n" if converged else "End\n"
    (run_dir / "log.foamRun").write_text("ExecutionTime = 8 s\n\n" + end)


def test_read_pressure_drop(tmp_path):
    _fake_log(tmp_path, True)
    _fake_patch_average(tmp_path, "inlet", {0: [(1, 5.0), (2, 13.0)]})
    _fake_patch_average(tmp_path, "outlet", {0: [(1, 0.0), (2, 1.0)]})
    assert read_pressure_drop(tmp_path, rho=1.2) == pytest.approx(1.2 * 12.0)


def test_read_pressure_drop_uses_latest_restart(tmp_path):
    _fake_log(tmp_path, True)
    _fake_patch_average(tmp_path, "inlet", {0: [(1, 5.0)], 100: [(101, 20.0)]})
    _fake_patch_average(tmp_path, "outlet", {0: [(1, 0.0)], 100: [(101, 0.0)]})
    assert read_pressure_drop(tmp_path, rho=1.0) == pytest.approx(20.0)


def test_read_pressure_drop_missing(tmp_path):
    _fake_log(tmp_path, True)
    with pytest.raises(FileNotFoundError):
        read_pressure_drop(tmp_path, rho=1.0)


def test_not_converged(tmp_path):
    _fake_log(tmp_path, False)
    _fake_patch_average(tmp_path, "inlet", {0: [(1, 5.0)]})
    _fake_patch_average(tmp_path, "outlet", {0: [(1, 0.0)]})
    assert not has_converged(tmp_path)
    with pytest.raises(NotConvergedError):
        read_pressure_drop(tmp_path, rho=1.0)
    assert read_pressure_drop(tmp_path, rho=1.0, require_converged=False) == pytest.approx(5.0)
