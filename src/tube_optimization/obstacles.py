import itertools
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from .stl import read_stl, write_stl


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


class MeshObstacle(Obstacle):
    """Obstacle given by an arbitrary closed triangle surface, e.g. modelled in a CAD
    program and exported as STL.

    Checking a point against thousands of triangles is slow and the optimizer does
    it all the time, so the signed distance is computed once on a regular grid
    when the obstacle is created (about a second for the default resolution).
    After that every query is a trilinear interpolation, with a cost that does not
    depend on the number of triangles. The interpolated field is continuous,
    which is what the optimizer needs from a constraint.

    The surface must be closed (watertight) to tell inside from outside. If it is
    not, the distance is still right, but it is positive everywhere.

    The accuracy is about one grid cell. Choose cell_size around the tube radius
    divided by 4 or smaller, or leave it to None for a grid of 100 cells along
    the largest dimension of the obstacle.

    surface: path to a file or an array of triangles (n, 3, 3). STL is read
    natively. Other formats (OBJ, PLY, OFF...) are read with trimesh if installed.
    scale and translate are applied to the vertices in this order, for example
    scale=0.001 turns a model in millimeters into meters.
    """

    MAX_CELLS = 40_000_000

    def __init__(self, surface, scale=1.0, translate=(0.0, 0.0, 0.0), cell_size=None, padding=None):
        if isinstance(surface, (str, Path)):
            tris = self._read(Path(surface))
        else:
            tris = np.asarray(surface, dtype=float)
        if tris.ndim != 3 or tris.shape[1:] != (3, 3) or len(tris) == 0:
            raise ValueError("the surface must be a non-empty array of triangles with shape (n, 3, 3)")
        self._triangles = tris * scale + np.asarray(translate, dtype=float)

        lo, hi = self._triangles.reshape(-1, 3).min(axis=0), self._triangles.reshape(-1, 3).max(axis=0)
        extent = np.max(hi - lo)
        if extent <= 0:
            raise ValueError("the surface has no volume")
        self.cell_size = cell_size if cell_size is not None else extent / 100
        # Far from the surface the exact value does not matter, only that it is large
        self.padding = padding if padding is not None else max(10 * self.cell_size, 0.1 * extent)
        self._build_field(lo - self.padding, hi + self.padding)

    @staticmethod
    def _read(path):
        if path.suffix.lower() == ".stl":
            return read_stl(path)
        try:
            import trimesh
        except ImportError:
            raise ImportError(f"{path.suffix} files need the trimesh package, or use an STL") from None
        mesh = trimesh.load(path, force="mesh")
        return np.asarray(mesh.triangles, dtype=float)

    def _surface_samples(self):
        """Points on the surface, at most cell_size / 4 apart."""
        spacing = self.cell_size / 4
        tris = self._triangles
        edges = np.stack([tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]], axis=1)
        n_div = np.maximum(1, np.ceil(np.linalg.norm(edges, axis=2).max(axis=1) / spacing).astype(int))
        samples = [tris.reshape(-1, 3)]
        for n in np.unique(n_div):     # triangles with the same subdivision share the lattice
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = i + j <= n
            u, v = i[keep] / n, j[keep] / n
            group = tris[n_div == n]
            samples.append((group[:, None, 0] + u[None, :, None] * (group[:, 1] - group[:, 0])[:, None]
                            + v[None, :, None] * (group[:, 2] - group[:, 0])[:, None]).reshape(-1, 3))
        return np.concatenate(samples)

    def _build_field(self, lo, hi):
        h = self.cell_size
        shape = np.ceil((hi - lo) / h).astype(int) + 1
        if np.prod(shape, dtype=float) > self.MAX_CELLS:
            raise ValueError(f"the grid would have {np.prod(shape):.3g} cells, use a larger cell_size")
        axes = [lo[k] + h * np.arange(shape[k]) for k in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

        samples = self._surface_samples()

        # Cells crossed by the surface form a closed wall, thanks to the sample
        # spacing. Everything that cannot be reached from the corner without
        # crossing it is inside.
        wall = np.zeros(shape, dtype=bool)
        idx = np.clip(np.round((samples - lo) / h).astype(int), 0, shape - 1)
        wall[idx[:, 0], idx[:, 1], idx[:, 2]] = True

        # Distance to the surface. Near it, exact (to the samples). A KD-tree is
        # very slow for points far from a hollow surface, since they are almost
        # equally far from all the samples, so there the distance to the wall
        # cells is used instead, which is only off by about one cell.
        band = 5 * h
        approx = ndimage.distance_transform_edt(~wall) * h
        distance = np.maximum(approx, band)
        near = approx < band + 2 * h
        exact = cKDTree(samples).query(grid[near], distance_upper_bound=band)[0]
        distance[near] = np.where(np.isfinite(exact), exact, distance[near])

        labels, _ = ndimage.label(~wall)
        inside = (labels != labels[0, 0, 0]) & ~wall
        if not inside.any():
            warnings.warn("the surface does not seem to be closed, the obstacle has no inside: "
                          "the distance to it is positive everywhere", stacklevel=3)

        self._lo, self._hi = lo, lo + h * (shape - 1)
        self._field = RegularGridInterpolator(axes, np.where(inside, -distance, distance))

    def signed_distance(self, points):
        points = np.atleast_2d(points)
        # Outside the grid, the distance of the closest point of the grid plus the gap
        inside_box = np.clip(points, self._lo, self._hi)
        return self._field(inside_box) + np.linalg.norm(points - inside_box, axis=1)

    def triangles(self):
        return self._triangles


def write_obstacles_stl(obstacles, path):
    """One STL with a named solid per obstacle (obstacle_0, obstacle_1, ...)."""
    write_stl(path, {f"obstacle_{i}": o.triangles() for i, o in enumerate(obstacles)})
