import numpy as np


class BezierCurve:
    """Bezier curve of arbitrary degree defined by its control points.

    The degree is just len(control_points) - 1, so two points give a straight
    line, three a quadratic curve, and so on.
    """

    def __init__(self, control_points):
        pts = np.asarray(control_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("control_points must have shape (n, 3) with n >= 2")
        self.control_points = pts

    @property
    def degree(self):
        return len(self.control_points) - 1

    def evaluate(self, t):
        """Points of the curve at the parameters t in [0, 1]. Returns (len(t), 3)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self._de_casteljau(self.control_points, t)

    def derivative(self, t, order=1):
        """Derivative of the given order at the parameters t. Returns (len(t), 3)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if order > self.degree:
            return np.zeros((len(t), 3))
        # The k-th derivative of a Bezier curve is another Bezier curve of k
        # degrees less, built from the k-th differences of the control points.
        scale = np.prod(np.arange(self.degree, self.degree - order, -1))
        pts = scale * np.diff(self.control_points, n=order, axis=0)
        return self._de_casteljau(pts, t)

    def tangent(self, t):
        """Unit tangent vectors at the parameters t in [0, 1]. Returns (len(t), 3)."""
        d = self.derivative(t, 1)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        if np.any(norm < 1e-12):
            raise ValueError("curve has a zero-velocity point, tangent is undefined")
        return d / norm

    def curvature(self, t):
        """Curvature |r' x r''| / |r'|^3 at the parameters t. Returns (len(t),)."""
        d1 = self.derivative(t, 1)
        d2 = self.derivative(t, 2)
        speed = np.linalg.norm(d1, axis=1)
        if np.any(speed < 1e-12):
            raise ValueError("curve has a zero-velocity point, curvature is undefined")
        return np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3

    @staticmethod
    def _de_casteljau(points, t):
        # Repeated linear interpolation, vectorised over all the t values
        work = np.broadcast_to(points[:, None, :], (len(points), len(t), 3)).copy()
        tt = t[:, None]
        for level in range(len(points) - 1):
            work = (1 - tt) * work[:-1] + tt * work[1:]
        return work[0]
