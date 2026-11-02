import numpy as np

from .bezier import BezierCurve
from .stl import write_stl


class TubeSelfIntersectionError(ValueError):
    """Raised when the tube would cross itself or fold on a tight bend."""


class TubeSurface:
    """Triangulated surface of a constant-radius tube following a Bezier curve.

    The surface is split in three patches, which are what the OpenFOAM case
    needs: "inlet" (t = 0 cap), "outlet" (t = 1 cap) and "tube" (lateral wall).
    All the normals point outwards, away from the fluid.
    """

    def __init__(self, curve, radius, n_axial=100, n_circ=32):
        if not isinstance(curve, BezierCurve):
            raise TypeError("curve must be a BezierCurve")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.curve = curve
        self.radius = radius
        self.n_axial = n_axial
        self.n_circ = n_circ

    def _rings(self):
        """Vertices of the cross-section circles, shape (n_axial, n_circ, 3)."""
        t = np.linspace(0.0, 1.0, self.n_axial)
        centers = self.curve.evaluate(t)
        tangents = self.curve.tangent(t)

        # Parallel transport of the frame along the curve, so the circles do not
        # twist around the axis (a Frenet frame breaks on straight sections).
        normals = np.zeros_like(tangents)
        normals[0] = self._any_perpendicular(tangents[0])
        for i in range(1, len(t)):
            n = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
            normals[i] = n / np.linalg.norm(n)
        binormals = np.cross(tangents, normals)

        theta = np.linspace(0.0, 2 * np.pi, self.n_circ, endpoint=False)
        cos, sin = np.cos(theta), np.sin(theta)
        offsets = (cos[None, :, None] * normals[:, None, :]
                   + sin[None, :, None] * binormals[:, None, :])
        return centers[:, None, :] + self.radius * offsets

    @staticmethod
    def _any_perpendicular(v):
        # Cross with the axis that is least aligned with v
        axis = np.zeros(3)
        axis[np.argmin(np.abs(v))] = 1.0
        p = np.cross(v, axis)
        return p / np.linalg.norm(p)

    def clearances(self):
        """Continuous margins of the geometric validity, both >= 0 for a valid tube.

        Returns (curvature_margin, wall_margin):
        - curvature_margin = 1 - max curvature * radius. Negative when the radius
          is larger than the radius of curvature, so the circles fold over each
          other on the inside of a bend.
        - wall_margin = distance between the two closest distant parts of the
          centerline minus one diameter. Negative when the walls overlap.
          Points closer than pi*r along the curve are neighbours, not a collision.

        Raises ValueError if the curve has a point where it stops.
        """
        t = np.linspace(0.0, 1.0, self.n_axial)
        r = self.radius

        curvature_margin = 1.0 - self.curve.curvature(t).max() * r

        centers = self.curve.evaluate(t)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))])
        dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        far = np.abs(arc[:, None] - arc[None, :]) > np.pi * r
        wall_margin = dist[far].min() - 2 * r if far.any() else np.inf
        return curvature_margin, wall_margin

    def check_self_intersection(self):
        """Raises TubeSelfIntersectionError if the tube is not geometrically valid.

        See clearances() for the two things that are checked.
        """
        r = self.radius
        curvature_margin, wall_margin = self.clearances()
        if curvature_margin <= 0:
            raise TubeSelfIntersectionError(
                f"radius {r} is larger than the minimum radius of curvature "
                f"{r / (1 - curvature_margin):.4g}")
        if wall_margin < 0:
            raise TubeSelfIntersectionError(
                f"tube walls overlap: closest approach is {wall_margin + 2 * r:.4g}, "
                f"minimum allowed is {2 * r:.4g}")

    def build(self):
        """Returns a dict {patch name: triangles}, triangles with shape (n, 3, 3)."""
        self.check_self_intersection()
        rings = self._rings()
        m = self.n_circ
        nxt = np.roll(np.arange(m), -1)  # index of the next vertex around the ring

        # Lateral wall: two triangles per quad between consecutive rings
        a = rings[:-1, :, :]
        b = rings[:-1, nxt, :]
        c = rings[1:, nxt, :]
        d = rings[1:, :, :]
        wall = np.concatenate([np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
                               np.stack([a, c, d], axis=2).reshape(-1, 3, 3)])

        inlet = self._cap(rings[0], self.curve.evaluate(0.0)[0], flip=True)
        outlet = self._cap(rings[-1], self.curve.evaluate(1.0)[0], flip=False)
        patches = {"inlet": inlet, "outlet": outlet, "tube": wall}

        # The winding order above depends on the frame handedness, so check the
        # wall orientation against the radial direction and fix all if needed.
        if self._wall_points_inwards(wall, rings):
            patches = {k: v[:, ::-1, :] for k, v in patches.items()}
        return patches

    @staticmethod
    def _cap(ring, center, flip):
        # Triangle fan from the center of the circle
        nxt = np.roll(np.arange(len(ring)), -1)
        tri = np.stack([np.broadcast_to(center, ring.shape), ring, ring[nxt]], axis=1)
        return tri[:, ::-1, :] if flip else tri

    @staticmethod
    def _wall_points_inwards(wall, rings):
        # Look at the first triangle: its normal should have a positive
        # component along the radial vector from the axis
        tri = wall[0]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        radial = tri.mean(axis=0) - rings[0].mean(axis=0)
        return np.dot(normal, radial) < 0

    def write_stl(self, path):
        """Writes an ASCII STL with one named solid per patch (OpenFOAM style)."""
        write_stl(path, self.build())
