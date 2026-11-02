import numpy as np

from .bezier import BezierCurve


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

    def build(self):
        """Returns a dict {patch name: triangles}, triangles with shape (n, 3, 3)."""
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
        with open(path, "w") as f:
            for name, tris in self.build().items():
                f.write(f"solid {name}\n")
                for tri in tris:
                    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    n = n / np.linalg.norm(n)
                    f.write(f"  facet normal {n[0]:e} {n[1]:e} {n[2]:e}\n")
                    f.write("    outer loop\n")
                    for v in tri:
                        f.write(f"      vertex {v[0]:e} {v[1]:e} {v[2]:e}\n")
                    f.write("    endloop\n  endfacet\n")
                f.write(f"endsolid {name}\n")
