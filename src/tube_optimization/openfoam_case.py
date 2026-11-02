import math
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from .tube_surface import TubeSurface

CASE_DIR = Path(__file__).resolve().parents[2] / "case"


def load_config(path=CASE_DIR / "config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


class OpenFOAMCase:
    """Builds an OpenFOAM case for a tube from the template and runs it.

    The template in case/template is a complete case where the values that
    change between designs are written as {{placeholders}}.
    """

    def __init__(self, tube, run_dir, config=None, template_dir=CASE_DIR / "template"):
        if not isinstance(tube, TubeSurface):
            raise TypeError("tube must be a TubeSurface")
        self.tube = tube
        self.run_dir = Path(run_dir)
        self.config = config if config is not None else load_config()
        self.template_dir = Path(template_dir)

    def create(self):
        """Copies the template, writes the STL and fills in the placeholders."""
        if self.run_dir.exists():
            raise FileExistsError(f"{self.run_dir} already exists")
        shutil.copytree(self.template_dir, self.run_dir)
        self.tube.write_stl(self.run_dir / "constant" / "triSurface" / "tube.stl")
        self._fill_placeholders(self._parameters())

    def run(self):
        """Runs the Allrun script of the case inside the distrobox container."""
        cfg = self.config["openfoam"]
        script = f"source {cfg['bashrc']} && cd '{self.run_dir.resolve()}' && ./Allrun"
        subprocess.run(["distrobox", "enter", cfg["container"], "--", "bash", "-lc", script],
                       check=True)

    def _parameters(self):
        flow, fluid = self.config["flow"], self.config["fluid"]
        mesh, solver = self.config["mesh"], self.config["solver"]
        r = self.tube.radius

        # Size of the background cells: the ones at the wall are radius / cells_per_radius
        # and every refinement level halves the size.
        level = mesh["surface_level"]
        dx = r / mesh["cells_per_radius"] * 2**level

        # Bounding box of the tube plus a margin, grown to a whole number of cubic cells
        pts = np.concatenate(list(self.tube.build().values())).reshape(-1, 3)
        lo = pts.min(axis=0) - mesh["margin_cells"] * dx
        n_cells = np.ceil((pts.max(axis=0) + mesh["margin_cells"] * dx - lo) / dx).astype(int)
        hi = lo + n_cells * dx

        # Inside point at the middle of the tube. A tiny offset keeps it away from
        # the faces of the background mesh.
        inside = self.tube.curve.evaluate(0.5)[0] + 1e-3 * dx

        # Turbulence values at the inlet from the intensity and a length scale
        u = flow["velocity"]
        k = 1.5 * (flow["turbulence_intensity"] * u) ** 2
        length = flow["length_scale_factor"] * 2 * r
        epsilon = 0.09**0.75 * k**1.5 / length

        params = {
            "xmin": lo[0], "ymin": lo[1], "zmin": lo[2],
            "xmax": hi[0], "ymax": hi[1], "zmax": hi[2],
            "nx": n_cells[0], "ny": n_cells[1], "nz": n_cells[2],
            "surface_level": level,
            "inside_x": inside[0], "inside_y": inside[1], "inside_z": inside[2],
            "nu": fluid["nu"], "velocity": u, "k": k, "epsilon": epsilon,
            "n_iterations": solver["n_iterations"],
        }
        return {name: f"{v:.8g}" if isinstance(v, (float, np.floating)) else str(v)
                for name, v in params.items()}

    def _fill_placeholders(self, params):
        for path in self.run_dir.rglob("*"):
            if not path.is_file() or path.suffix == ".stl":
                continue
            text = path.read_text()
            for name, value in params.items():
                text = text.replace("{{" + name + "}}", value)
            left = re.findall(r"\{\{\w+\}\}", text)
            if left:
                raise ValueError(f"unresolved placeholders in {path}: {left}")
            path.write_text(text)
