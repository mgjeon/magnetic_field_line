import numpy as np
import pyvista as pv
from matplotlib import colors

from streamtracer import VectorGrid, StreamTracer

__all__ = ["CartesianPlotter"]

class CartesianPlotter(pv.Plotter):
    """
    A plotter for 3D data in a Cartesian box.

    This class inherits `pyvista.Plotter`. It is used to visualize 3D vector field lines
    (e.g., magnetic field lines of solar active regions) traced by `streamtracer`.

    Parameters
    ----------
    kwargs : dict
        All other keyword arguments are passed through to `pyvista.Plotter`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_vector_field(self, vectors, *args, **kwargs):
        """
        Define a 3D vector field.

        Parameters
        ----------
        vectors : ndarray
            A (nx, ny, nz, 3) array representing the 3D vector field.
        args: list
            arguments for `streamtracer.VectorGrid`.
        kwargs: dict
            keyword arguments for `streamtracer.VectorGrid`.
        """
        if kwargs.get("grid_spacing") is None:
            if kwargs.get("grid_coords") is None:
                kwargs["grid_spacing"] = [1, 1, 1]
        self.grid = VectorGrid(vectors.astype(np.float64), *args, **kwargs)
        self.xcoords = self.grid.xcoords.astype(np.float64)
        self.ycoords = self.grid.ycoords.astype(np.float64)
        self.zcoords = self.grid.zcoords.astype(np.float64)
        self.x, self.y, self.z = np.meshgrid(self.xcoords, self.ycoords, self.zcoords, indexing="ij")

    def _create_mesh(self):
        """
        Create a `pyvista.StructuredGrid` mesh from the 3D vector field.
        """
        mesh = pv.StructuredGrid(self.x, self.y, self.z)
        vectors = self.grid.vectors.transpose(2, 1, 0, 3).reshape(-1, 3)
        mesh["vectors"] = vectors
        mesh.active_vectors_name = "vectors"
        magnitudes = np.linalg.norm(vectors, axis=-1)
        mesh["magnitudes"] = magnitudes
        mesh.active_scalars_name = "magnitudes"
        self.mesh = mesh

    def plot_outline(self, color="black", **kwargs):
        """
        Plot the outline of the 3D vector field.

        Parameters
        ----------
        color : str, optional
            The color of the outline.
            Default is 'black'.
        kwargs : dict
            Keyword arguments for `pyvista.Plotter.add_mesh`.
        """
        self._create_mesh()
        self.add_mesh(self.mesh.outline(), color=color, **kwargs)

    def plot_boundary(self, boundary="bottom", component=2, cmap="gray", **kwargs):
        """
        Plot the boundary of the 3D vector field.

        Parameters
        ----------
        boundary : str, optional
            The boundary to be plotted.
            'bottom', 'top', 'left', 'right', 'front', or 'back'.
            Default is 'bottom'.
        component : int, optional
            The component of the vector field to be plotted.
            0, 1, or 2 for x, y, or z component, respectively.
            Default is 2.
        cmap : str, optional
            The colormap for the boundary.
            Default is 'gray'.
        kwargs : dict
            Keyword arguments for `pyvista.Plotter.add_mesh`.
        """
        self._create_mesh()
        nx, ny, nz = self.mesh.dimensions
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = nx - 1, ny - 1, nz - 1

        if boundary == "bottom":
            subset = (x_min, x_max, y_min, y_max, z_min, z_min)
        elif boundary == "top":
            subset = (x_min, x_max, y_min, y_max, z_max, z_max)
        elif boundary == "left":
            subset = (x_min, x_min, y_min, y_max, z_min, z_max)
        elif boundary == "right":
            subset = (x_max, x_max, y_min, y_max, z_min, z_max)
        elif boundary == "front":
            subset = (x_min, x_max, y_min, y_min, z_min, z_max)
        elif boundary == "back":
            subset = (x_min, x_max, y_max, y_max, z_min, z_max)
        else:
            raise ValueError("Invalid boundary. Choose from 'bottom', 'top', 'left', 'right', 'front', or 'back'.")

        surface = self.mesh.extract_subset(subset).extract_surface()
        surface.active_vectors_name = "vectors"
        surface.active_scalars_name = "magnitudes"
        self.add_mesh(surface, scalars="vectors", component=component, cmap=cmap, lighting=False, **kwargs)

    def plot_field_lines(
        self,
        seeds,
        render_lines_as_tubes=True,
        radius=1,
        max_steps=10000,
        step_size=0.1,
        seeds_config=dict(show_seeds=False, color="red", point_size=5),
        **kwargs,
    ):
        """
        Plot field lines traced from seeds using `streamtracer.StreamTracer`.

        Parameters
        ----------
        seeds : ndarray
            A (N, 3) array representing the seeds.
        render_lines_as_tubes : bool, optional
            Whether to render field lines as tubes.
            Default is True.
        radius : float, optional
            The radius of the tubes for rendering field lines.
            Default is 1.
        max_steps : int, optional
            The maximum number of steps for tracing field lines.
            Default is 10000.
        step_size : float, optional
            The step size for tracing field lines.
            Default is 0.1.
        seeds_config : dict, optional
            Configuration for plotting seeds.
            Default is dict(show_seeds=False, color='red', point_size=5).
        kwargs : dict
            Keyword arguments for `pyvista.Plotter.add_mesh`.
        """
        tracer = StreamTracer(max_steps, step_size)
        tracer.trace(seeds, self.grid)
        tracer_xs = []
        tracer_xs.append(tracer.xs)
        tracer_xs = [item for sublist in tracer_xs for item in sublist]
        for i, xl in enumerate(tracer_xs):
            assert seeds[i] in xl
            if len(xl) < 2:
                continue
            spline = pv.Spline(xl)
            if render_lines_as_tubes:
                spline = spline.tube(radius=radius)
            self.add_mesh(spline, **kwargs)
        if seeds_config.get("show_seeds"):
            seeds_config.pop("show_seeds")
            self.add_mesh(pv.PolyData(seeds), **seeds_config)