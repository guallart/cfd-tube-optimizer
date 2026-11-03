import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .bezier import BezierCurve
from .obstacles import write_obstacles_stl
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

    Obstacles are regions the tube cannot enter: the centerline has to stay
    further than the tube radius from each of them. They are not part of the
    CFD. The geometry (obstacles, self intersection, minimum segment length) is
    handled as constraints, not as a penalty on the pressure drop: they are
    cheap to evaluate and continuous, so COBYLA sees them as such. When the
    optimizer proposes a design that violates them, it is moved back towards
    the best valid design until it is valid, so no CFD run is wasted on it.

    Design variables x, all in units of the inlet-outlet distance L:
      x[0], x[1]  distance of the 2nd and the 2nd to last control points to the ports
      x[2:]       (dx, dy, dz) of every free control point, measured from its
                  position on the straight line between the ports
    """

    SOLVERS = ("cobyla", "cma-es")
    MIN_SEGMENT = 0.05    # in units of L, shorter segments make the path fold back at the ports

    MAX_START_ATTEMPTS = 10   # starting designs tried before giving up

    # Rows of the history that did not give a pressure drop are marked with these
    INVALID = ("short_segment", "self_intersection", "obstacle", "not_converged",
               "openfoam_error")

    def __init__(self, inlet_point, inlet_direction, outlet_point, outlet_direction,
                 radius, degree=5, obstacles=(), results_dir="results/optimization",
                 config=None, seed=0, solver=None):
        if degree < 3:
            raise ValueError("degree must be at least 3 to impose both end directions")
        self.p_in = np.asarray(inlet_point, dtype=float)
        self.p_out = np.asarray(outlet_point, dtype=float)
        self.d_in = self._unit(inlet_direction)
        self.d_out = self._unit(outlet_direction)
        self.radius = radius
        self.degree = degree
        self.obstacles = list(obstacles)
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.config = config if config is not None else load_config()
        # Solver and its settings come from the [optimizer] table of config.toml
        options = self.config.get("optimizer", {})
        self.solver = solver or options.get("solver", "cma-es")
        if self.solver not in self.SOLVERS:
            raise ValueError(f"solver must be one of {self.SOLVERS}, not {self.solver!r}")
        self.initial_step = options.get("initial_step", 0.1)   # in units of L
        self.x_tolerance = options.get("x_tolerance", 0.01)    # in units of L

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

    def initial_guess(self, attempt=0):
        """Starting design: a smooth cubic-like path if it is valid, otherwise the
        shortest valid one out of a batch of random paths. A different attempt
        number gives a different random batch."""
        x = np.zeros(2 + 3 * self.n_free)
        x[:2] = 1 / 3
        if attempt == 0 and self._geometry_status(x)[1] == "ok":
            return x

        # The default path crosses an obstacle (or itself). Looking for a valid
        # one is only geometry, so it costs no simulations.
        rng = np.random.default_rng(self.seed + attempt)
        best, best_length = None, np.inf
        t = np.linspace(0, 1, 100)
        for _ in range(2000):
            cand = np.concatenate([rng.uniform(0.1, 1.0, 2), rng.normal(0, 0.5, 3 * self.n_free)])
            if self._geometry_status(cand)[1] != "ok":
                continue
            pts = self.build_curve(cand).evaluate(t)
            length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            if length < best_length:
                best, best_length = cand, length
        if best is None:
            raise RuntimeError("could not find a valid starting path, try a higher degree")
        return best

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
        """Runs the configured solver from the initial guess and returns an OptimizationResult."""
        if self.results_dir.exists() and any(self.results_dir.iterdir()):
            raise FileExistsError(f"{self.results_dir} is not empty")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.obstacles:
            write_obstacles_stl(self.obstacles, self.results_dir / "obstacles.stl")

        # A starting design whose simulation fails is discarded and another one is tried
        for attempt in range(self.MAX_START_ATTEMPTS):
            x0 = self.initial_guess(attempt)
            dp0 = self.evaluate(x0)
            if dp0 is not None:
                break
        else:
            raise RuntimeError(f"no valid initial design after {self.MAX_START_ATTEMPTS} attempts, "
                               "see history.csv")
        # Only for simulations that fail (not converged, OpenFOAM error)
        self._penalty = 10 * dp0

        if self.solver == "cobyla":
            self._run_cobyla(x0, max_evaluations)
        else:
            self._run_cma_es(x0, dp0, max_evaluations)

        dp, x, run_dir = self._best
        return OptimizationResult(self.build_curve(x), dp, x, run_dir, self._n_evals)

    def _run_cobyla(self, x0, max_evaluations):
        minimize(self._objective, x0, method="COBYLA",
                 constraints=[{"type": "ineq", "fun": self.constraints}],
                 options={"rhobeg": self.initial_step, "tol": self.x_tolerance,
                          "maxiter": max_evaluations})

    def _run_cma_es(self, x0, dp0, max_evaluations):
        """CMA-ES: samples a population around a mean, keeps the best and adapts the
        covariance. Robust to the CFD noise and to flat regions, at the price of
        needing more evaluations than COBYLA.

        The sampled designs that are not valid are moved to the closest valid
        one, and CMA-ES is told about the moved design (not the sampled one), so
        the distribution does not drift into the forbidden region.
        """
        import cma    # only needed by this solver

        es = cma.CMAEvolutionStrategy(x0, self.initial_step, {
            "seed": self.seed + 1,             # 0 would mean a random seed
            "maxfevals": max_evaluations,
            "tolx": self.x_tolerance * self.initial_step,
            "tolfun": 0.005 * dp0,             # CFD noise is around 1%
            "verbose": -9,
        })
        while not es.stop() and self._n_evals < max_evaluations:
            candidates = [self._repair(x) for x in es.ask()]
            values = [self._objective(x) for x in candidates]
            es.tell(candidates, values)

    def constraints(self, x):
        """Geometric constraints of the design x, all >= 0 when it is valid.

        Lengths are divided by the inlet-outlet distance, so every entry is of order 1.
        Entries: the two end segments, curvature, wall overlap, one per obstacle.
        """
        x = np.asarray(x, dtype=float)
        segments = (x[:2] - self.MIN_SEGMENT) / self.MIN_SEGMENT
        try:
            tube = TubeSurface(self.build_curve(x), self.radius)
            curvature_margin, wall_margin = tube.clearances()
            centerline = tube.curve.evaluate(np.linspace(0, 1, 300))
            clearance = [(o.signed_distance(centerline).min() - self.radius) / self.length
                         for o in self.obstacles]
        except ValueError:   # a point where the curve stops
            curvature_margin = wall_margin = -1.0
            clearance = [-1.0] * len(self.obstacles)
        return np.concatenate([segments, [curvature_margin, min(wall_margin / self.length, 1.0)],
                               clearance])

    def _repair(self, x):
        """Returns x if it is valid, otherwise the closest valid design.

        Only geometry is involved, so this costs no simulations. A first valid
        point is found by bisection towards the best valid design so far, and
        then SLSQP moves it to the valid point closest to x.
        """
        x = np.asarray(x, dtype=float)
        if self._is_valid(x):
            return x
        anchor = self._best[1]
        lo, hi = 0.0, 1.0     # fraction of the way from the anchor to x
        for _ in range(25):
            mid = (lo + hi) / 2
            if self._is_valid(anchor + mid * (x - anchor)):
                lo = mid
            else:
                hi = mid
        start = anchor + lo * (x - anchor)

        projection = minimize(
            lambda z: np.sum((z - x) ** 2), start, jac=lambda z: 2 * (z - x), method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda z: self.constraints(z) - 1e-3}],
            options={"maxiter": 30, "ftol": 1e-8})
        if self._is_valid(projection.x) and np.sum((projection.x - x) ** 2) < np.sum((start - x) ** 2):
            return projection.x
        return start

    def _is_valid(self, x):
        return bool(np.all(self.constraints(x) >= 0))

    def _objective(self, x):
        dp = self.evaluate(self._repair(x))
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
        tube, status = self._geometry_status(x)
        if status != "ok":
            return None, status
        try:
            return self._simulate(tube, run_dir), "ok"
        except NotConvergedError:
            return None, "not_converged"
        except subprocess.CalledProcessError:
            return None, "openfoam_error"

    def _geometry_status(self, x):
        """Checks a design without simulating it. Returns (tube, "ok") or (None, reason)."""
        g = self.constraints(x)
        if g[:2].min() < 0:
            return None, "short_segment"
        if g[2:4].min() < 0:
            return None, "self_intersection"
        if g[4:].size and g[4:].min() < 0:
            return None, "obstacle"
        return TubeSurface(self.build_curve(x), self.radius), "ok"

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
