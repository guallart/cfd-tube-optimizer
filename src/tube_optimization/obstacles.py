import itertools

import numpy as np

from .stl import write_stl


class Obstacle:
    """A region of space the tube is not allowed to enter.

    Subclasses give a signed distance (negative inside) and a triangulated
    surface, which is only used to look at the obstacle in ParaView.
    """

    def signed_distance(self, points):
        """Distance from each point (n, 3) to the obstacle surface, negative inside."""
        raise NotImplementedError

    def triangles(self):
        """Surface triangles with outward normals, shape (n, 3, 3)."""
        raise NotImplementedError

    @staticmethod
    def _orient_outwards(tris, center):
        # Both shapes are convex, so a triangle is oriented outwards when its
        # normal points away from the center. Swap two vertices otherwise.
        tris = np.array(tris, dtype=float)
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        outward = np.einsum("ij,ij->i", normals, tris.mean(axis=1) - center)
        tris[outward < 0] = tris[outward < 0][:, ::-1]
        return tris


class Sphere(Obstacle):
    def __init__(self, center, radius, n_lat=24, n_lon=48):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.n_lat = n_lat
        self.n_lon = n_lon

    def signed_distance(self, points):
        return np.linalg.norm(np.atleast_2d(points) - self.center, axis=1) - self.radius

    def triangles(self):
        phi = np.linspace(0, np.pi, self.n_lat + 1)
        theta = np.linspace(0, 2 * np.pi, self.n_lon, endpoint=False)
        p, t = np.meshgrid(phi, theta, indexing="ij")
        v = self.center + self.radius * np.stack(
            [np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)], axis=-1)

        tris = []
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                j2 = (j + 1) % self.n_lon
                a, b, c, d = v[i, j], v[i + 1, j], v[i + 1, j2], v[i, j2]
                # At the poles two corners coincide, skip the flat triangle
                if i > 0:
                    tris.append([a, b, d])
                if i < self.n_lat - 1:
                    tris.append([b, c, d])
        return self._orient_outwards(tris, self.center)


class Box(Obstacle):
    """Axis-aligned box given by two opposite corners."""

    def __init__(self, corner_min, corner_max):
        self.lo = np.minimum(corner_min, corner_max).astype(float)
        self.hi = np.maximum(corner_min, corner_max).astype(float)
        if np.any(self.hi - self.lo <= 0):
            raise ValueError("the box must have a positive size in every direction")
        self.center = (self.lo + self.hi) / 2

    def signed_distance(self, points):
        q = np.abs(np.atleast_2d(points) - self.center) - (self.hi - self.lo) / 2
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(q.max(axis=1), 0)
        return outside + inside

    def triangles(self):
        bounds = (self.lo, self.hi)
        corners = np.array([[bounds[i][k] for k, i in enumerate(idx)]
                            for idx in itertools.product((0, 1), repeat=3)])
        # Corner index is 4*ix + 2*iy + iz, four corners per face
        faces = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        tris = []
        for a, b, c, d in faces:
            tris += [[corners[a], corners[b], corners[c]], [corners[a], corners[c], corners[d]]]
        return self._orient_outwards(tris, self.center)


def write_obstacles_stl(obstacles, path):
    """One STL with a named solid per obstacle (obstacle_0, obstacle_1, ...)."""
    write_stl(path, {f"obstacle_{i}": o.triangles() for i, o in enumerate(obstacles)})
