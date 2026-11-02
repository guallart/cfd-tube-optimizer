import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .bezier import BezierCurve
from .openfoam_case import NotConvergedError, OpenFOAMCase, load_config
from .tube_surface import TubeSurface


@dataclass
class OptimizationResult:
    curve: BezierCurve
    pressure_drop: float   # in Pa
    x: np.ndarray          # design variables, see TubeOptimizer.build_curve
    run_dir: Path          # OpenFOAM case of the best design
    n_evaluations: int     # number of designs tried (simulated or rejected)


class TubeOptimizer:
    """Finds the Bezier path between two fixed ports with the lowest pressure drop.

    The inlet and outlet positions are fixed, and so are the flow directions
    there (the direction the fluid moves when it enters and when it leaves).
    To respect them, the second control point lies on the line leaving the inlet
    and the second to last one on the line arriving at the outlet. The control
    points in between are free.

    Design variables x, all in units of the inlet-outlet distance L:
      x[0], x[1]  distance of the 2nd and the 2nd to last control points to the ports
      x[2:]       (dx, dy, dz) of every free control point, measured from its
                  position on the straight line between the ports
    """

    # Rows of the history that did not give a pressure drop are marked with these
    INVALID = ("short_segment", "self_intersection", "not_converged", "openfoam_error")

    def __init__(self, inlet_point, inlet_direction, outlet_point, outlet_direction,
                 radius, degree=5, results_dir="results/optimization", config=None):
        if degree < 3:
            raise ValueError("degree must be at least 3 to impose both end directions")
        self.p_in = np.asarray(inlet_point, dtype=float)
        self.p_out = np.asarray(outlet_point, dtype=float)
        self.d_in = self._unit(inlet_direction)
        self.d_out = self._unit(outlet_direction)
        self.radius = radius
        self.degree = degree
        self.results_dir = Path(results_dir)
        self.config = config if config is not None else load_config()

        self.length = np.linalg.norm(self.p_out - self.p_in)
        self.n_free = degree - 3
        self._cache = {}          # x -> pressure drop (or None if invalid)
        self._n_evals = 0
        self._penalty = None
        self._best = None         # (pressure drop, x, run_dir)

    @staticmethod
    def _unit(v):
        v = np.asarray(v, dtype=float)
        return v / np.linalg.norm(v)

    def initial_guess(self):
        """Smooth cubic-like path: end segments of a third of the distance, no offsets."""
        x = np.zeros(2 + 3 * self.n_free)
        x[:2] = 1 / 3
        return x

    def build_curve(self, x):
        x = np.asarray(x, dtype=float)
        L = self.length
        pts = [self.p_in, self.p_in + x[0] * L * self.d_in]
        for i, offset in enumerate(x[2:].reshape(-1, 3), start=2):
            on_line = self.p_in + (i / self.degree) * (self.p_out - self.p_in)
            pts.append(on_line + L * offset)
        pts += [self.p_out - x[1] * L * self.d_out, self.p_out]
        return BezierCurve(pts)

    def run(self, max_evaluations=100):
        """Runs Nelder-Mead from the initial guess and returns an OptimizationResult."""
        if self.results_dir.exists() and any(self.results_dir.iterdir()):
            raise FileExistsError(f"{self.results_dir} is not empty")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        x0 = self.initial_guess()
        dp0 = self.evaluate(x0)
        if dp0 is None:
            raise RuntimeError("the initial design is not valid, see history.csv")
        # Anything invalid is scored as much worse than the starting point
        self._penalty = 10 * dp0

        step = 0.1
        simplex = np.vstack([x0, x0 + step * np.eye(len(x0))])
        minimize(self._objective, x0, method="Nelder-Mead", options={
            "initial_simplex": simplex,
            "maxfev": max_evaluations,
            "xatol": 0.01,
            # CFD noise is around 1%, there is no point in resolving less than that
            "fatol": 0.005 * dp0,
        })

        dp, x, run_dir = self._best
        return OptimizationResult(self.build_curve(x), dp, x, run_dir, self._n_evals)

    def _objective(self, x):
        dp = self.evaluate(x)
        return self._penalty if dp is None else dp

    def evaluate(self, x):
        """Pressure drop of the design x in Pa, or None if the design is not valid."""
        key = np.asarray(x, dtype=float).tobytes()
        if key in self._cache:
            return self._cache[key]

        self._n_evals += 1
        run_dir = self.results_dir / f"eval_{self._n_evals:04d}"
        dp, status = self._try_design(np.asarray(x, dtype=float), run_dir)
        self._cache[key] = dp
        self._log(status, dp, x)

        if dp is not None and (self._best is None or dp < self._best[0]):
            self._best = (dp, np.array(x), run_dir)
        return dp

    def _try_design(self, x, run_dir):
        # Segments that are too short make the path fold back at the ports
        if x[0] < 0.05 or x[1] < 0.05:
            return None, "short_segment"
        try:
            tube = TubeSurface(self.build_curve(x), self.radius)
            tube.check_self_intersection()
        except ValueError:   # self intersection, or a point where the curve stops
            return None, "self_intersection"
        try:
            return self._simulate(tube, run_dir), "ok"
        except NotConvergedError:
            return None, "not_converged"
        except subprocess.CalledProcessError:
            return None, "openfoam_error"

    def _simulate(self, tube, run_dir):
        case = OpenFOAMCase(tube, run_dir, self.config)
        case.create()
        case.run()
        return case.pressure_drop()

    def _log(self, status, dp, x):
        shown = f"{dp:.4f} Pa" if dp is not None else "-"
        print(f"[{self._n_evals:4d}] {status:16s} {shown}", flush=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        path = self.results_dir / "history.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["eval", "status", "pressure_drop"]
                                + [f"x{i}" for i in range(len(x))])
            writer.writerow([self._n_evals, status, "" if dp is None else dp] + list(x))
