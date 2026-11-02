from tube_optimization import BezierCurve, OpenFOAMCase, TubeSurface


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
