import copy
import shutil

import pytest

from tube_optimization import BezierCurve, OpenFOAMCase, TubeSurface, load_config

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("distrobox") is None, reason="distrobox not available"),
]


def test_pressure_drop_mesh_independence(tmp_path):
    """The pressure drop must not change much when the mesh is refined.

    With 4 cells per radius the mesh is still too coarse (about 15% off), so the
    check starts at 6. The values are not monotonic with refinement, so each
    mesh is compared against the finest one instead of against the previous.
    """
    curve = BezierCurve([[0, 0, 0], [1, 0.5, 0], [2, -0.5, 0.3], [3, 0, 0]])
    tube = TubeSurface(curve, radius=0.1, n_axial=80)
    base = load_config()

    dp = {}
    for cpr in (6, 8, 12):
        cfg = copy.deepcopy(base)
        cfg["mesh"]["cells_per_radius"] = cpr
        case = OpenFOAMCase(tube, tmp_path / f"cpr{cpr}", cfg)
        case.create()
        case.run()
        dp[cpr] = case.pressure_drop()  # raises if the run did not converge

    reference = dp[12]
    for cpr in (6, 8):
        assert dp[cpr] == pytest.approx(reference, rel=0.05), dp
