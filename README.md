# Tube optimization

I wanted to know what shape a tube should have to move a fluid from one point to another with as little pressure loss as possible. The tube has a fixed radius, its two ends are fixed in space (position and flow direction), and there may be obstacles in the way that it has to go around.

![Optimization animation](assets/opt.GIF)

The path of the tube is a Bézier curve of any degree. An optimizer moves its control points, and for every candidate the pressure drop is computed with OpenFOAM. So each iteration of the optimization is a full CFD run: build the surface, mesh it, solve, read the pressure.

## What the solver does

The fluid is treated as incompressible and the flow as steady and turbulent. With the default settings (air, 10 m/s, 0.1 m radius) the Reynolds number, $\mathrm{Re} = UD/\nu$, is around 130,000, so laminar flow is not an option.

The case is solved with the Reynolds-averaged Navier-Stokes equations: mass conservation,

$$\nabla \cdot \mathbf{u} = 0,$$

and momentum,

$$\nabla \cdot (\mathbf{u}\,\mathbf{u}) = -\nabla p + \nabla \cdot \left[ (\nu + \nu_t) \left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right) \right],$$

where $p$ is the *kinematic* pressure (pressure divided by density, which is how OpenFOAM works with incompressible flows) and $\nu_t$ is the turbulent viscosity. That $\nu_t$ comes from the standard $k$-$\varepsilon$ model,

$$\nu_t = C_\mu \frac{k^2}{\varepsilon}, \qquad C_\mu = 0.09,$$

with two extra transport equations for the turbulent kinetic energy $k$ and its dissipation rate $\varepsilon$. Pressure and velocity are coupled with SIMPLE (the consistent variant, SIMPLEC), iterating until the residuals drop below a threshold. It is a steady solution, there is no time stepping.

Boundary conditions:

- **Inlet**: uniform velocity $U$, normal to the surface. $k$ and $\varepsilon$ are estimated from a turbulence intensity $I$ (5% by default) and a length scale $\ell$ of 7% of the diameter $D$, which is the usual rule of thumb:

  $$k = \frac{3}{2} (I\,U)^2, \qquad \varepsilon = C_\mu^{3/4} \frac{k^{3/2}}{\ell}, \qquad \ell = 0.07\,D$$

- **Outlet**: $p = 0$, zero gradient for everything else.
- **Wall**: no slip, with wall functions for $k$, $\varepsilon$ and $\nu_t$. I don't resolve the boundary layer, the first cell is assumed to sit in the logarithmic region.

The pressure drop is the area-averaged static pressure at the inlet minus the one at the outlet, multiplied by the density to get pascals:

$$\Delta P = \rho \left( \overline{p}_{\mathrm{in}} - \overline{p}_{\mathrm{out}} \right)$$

Static, not total: since both ends have the same area the difference in dynamic pressure is small, but it is not exactly zero because the velocity profile develops along the tube.

The mesh is made with snappyHexMesh, a hex-dominant mesh refined at the wall, with no boundary layer cells. They tend to fail on tight curves and I preferred a mesh that always works.

### How much to trust the numbers

For the test tube I used, the pressure drop changes about 1.5% between the default mesh and one twice as fine, so mesh resolution is not the main problem. The noise is: moving a control point a tiny bit changes the mesh, and with it the result, by 1 or 2%. That is why the optimizer is derivative-free (COBYLA or CMA-ES, see below) and why improvements smaller than that mean nothing. Because it starts from a single point it can also get stuck in a local minimum.

Besides spheres and boxes, an obstacle can be any closed surface modelled in other software and loaded from an STL: `MeshObstacle("part.stl", scale=0.001, translate=[x, y, z])` (`scale` converts the units of the model to meters). Checking the tube against thousands of triangles on every design would be slow, so the signed distance to the surface is computed once on a grid when the file is loaded, and each check afterwards is an interpolation that does not depend on the number of triangles. The accuracy is about one grid cell (by default 1% of the size of the obstacle; pass `cell_size` to change it, around a quarter of the tube radius or less is a good value). The surface has to be closed, otherwise a warning is shown. Other formats (OBJ, PLY...) are read if `trimesh` is installed.

Obstacles are only forbidden regions for the tube. They are not part of the CFD and do not disturb the flow. The geometry (obstacles, self intersection, minimum end segments) is handled as constraints, not as a penalty on the pressure drop: it is cheap to evaluate, so a design that violates it is first moved to the closest valid one and only that one is simulated. No CFD run is spent on an invalid tube.

## Using it

You need Python 3, an OpenFOAM 14 installation reachable through a distrobox container (the name and the path to the environment are in `case/config.toml`), and ParaView and ffmpeg if you want the animation.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The solver is chosen in the `[optimizer]` table of `case/config.toml` (default: `cma-es`):

- `cobyla`: local, uses few evaluations. Good when the starting design is already reasonable.
- `cma-es`: samples a population of designs per generation and adapts to the landscape. It tolerates the CFD noise and flat regions better and explores more, but needs more evaluations (use a larger `--max-evaluations`).

Both handle the geometric constraints in the same way. `initial_step` and `x_tolerance` in the same table set the size of the first step and when to stop.

Edit the top of `run_optimization.py` (ports, radius, degree of the curve, obstacles) and run it:

```bash
.venv/bin/python run_optimization.py --max-evaluations 100
```

Everything ends up in `results/optimization_<date>/`: one OpenFOAM case per design (open `case.foam` in ParaView), the obstacles as an STL, a `history.csv` and the best design in `best.json`. Fluid properties, velocity, mesh resolution and the number of parallel processes are in `case/config.toml`.

To get a video of the whole optimization: the tube of each iteration colored by its wall pressure (one fixed color scale for all iterations, or `--pressure-range MIN MAX`), with the pressure drop history floating in the upper right corner. The camera slowly turns around the scene at a constant speed (`--rotation-speed`, degrees per second, 12 by default, 0 for a fixed camera), independent of the iterations:

```bash
pvbatch animate_results.py results/optimization_<date>
```

Tests: `.venv/bin/pytest` runs the fast ones. `.venv/bin/pytest -m slow` also runs real simulations, and takes about a minute.
