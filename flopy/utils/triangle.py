from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..utils.cvfdutil import centroid_of_polygon
from ..utils.geospatial_utils import GeoSpatialUtil
from .utl_import import import_optional_dependency


class Triangle:
    """
    Class to work with the triangle program to unstructured triangular grids.
    Information on the triangle program can be found at
    https://www.cs.cmu.edu/~quake/triangle.html

    Parameters
    ----------
    model_ws : str
        workspace location for creating triangle files (default is '.')
        Deprecated.
    exe_name : str
        path and name of the triangle program. (default is triangle, which
        means that the triangle program must be in your path).
        Deprecated.
    maximum_area : float
        the maximum area for any triangle.  The default value is None, which
        means that the user must specify maximum areas for each region.
    angle : float
        Triangle will continue to add vertices until no angle is less than
        this specified value.  (default is 20 degrees)
    nodes : ndarray
        Two dimensional array of shape (npoints, 2) with x and y positions
        of fixed node locations to include in the resulting triangular mesh.
        (default is None)
    additional_args : list
        list of additional command line switches to pass to triangle

    Returns
    -------
    None

    """

    def __init__(
        self,
        model_ws=".",
        exe_name="triangle",
        maximum_area=None,
        angle=20.0,
        nodes=None,
        additional_args=None,
    ):
        self.angle = angle
        self.maximum_area = maximum_area
        self._nodes = nodes
        self.additional_args = additional_args
        self._initialize_vars()

    def add_polygon(self, polygon, ignore_holes=False):
        """
        Add a polygon

        Parameters
        ----------
        polygon : list, geojson, shapely.geometry, shapefile.Shape
            add polygon method accepts any of these geometries:

            a list of (x, y) points
            geojson Polygon object
            shapely Polygon object
            shapefile Polygon shape
            flopy.utils.geometry.Polygon object
        ignore_holes : bool
            method to ignore holes in polygon and only use the exterior
            coordinates

        Returns
        -------
        None

        """
        if isinstance(polygon, (list, tuple, np.ndarray)):
            polygon = [polygon]

        geom = GeoSpatialUtil(polygon, shapetype="Polygon")
        polygon = geom.points
        if polygon[0][0] == polygon[0][-1]:
            polygon[0] = polygon[0][:-1]
        self._polygons.append(polygon[0])
        if not ignore_holes:
            if len(polygon) > 1:
                for hole in polygon[1:]:
                    self.add_hole(hole)

    def add_hole(self, hole):
        """
        Add a point that will turn enclosing polygon into a hole

        Parameters
        ----------
        hole : tuple
            (x, y)

        Returns
        -------
        None

        """
        self._holes.append(hole)

    def add_region(self, point, attribute=0, maximum_area=None):
        """
        Add a point that will become a region with a maximum area, if
        specified.

        Parameters
        ----------
        point : tuple
            (x, y)

        attribute : integer or float
            integer value assigned to output elements

        maximum_area : float
            maximum area of elements in region

        Returns
        -------
        None

        """
        point = GeoSpatialUtil(point, shapetype="point").points
        self._regions.append([point, attribute, maximum_area])

    def build(self, verbose=False):
        """
        Build the triangular mesh

        Parameters
        ----------
        verbose : bool
            If true, print the results of the triangle command to the terminal
            (default is False)

        Returns
        -------
        None

        """

        import_optional_dependency(
            "triangle",
            error_message="Triangle requires triangle ().",
        )
        from triangle import triangulate  # python wrapper for the triangle lib (cython)

        # Construct the triangle command
        cmds: List[str] = []
        if self.maximum_area is not None:
            cmds.append(f"-a{self.maximum_area}")
        else:
            cmds.append("-a")
        if self.angle is not None:
            cmds.append(f"-q{self.angle}")
        if self.additional_args is not None:
            cmds += self.additional_args
        cmds.append("-A")  # assign attributes
        cmds.append("-p")  # triangulate .poly file
        if verbose:
            cmds.append("-V")  # verbose
        cmds.append("-D")  # delaunay triangles for finite volume
        cmds.append("-e")  # edge file
        cmds.append("-n")  # neighbor file

        t = triangulate(self._get_triangulate_input(), " ".join(cmds))

        # vertices
        self.vertices = t.get("vertices")
        self.vertex_markers = t.get("vertex_markers")
        self.edges = t.get("edges")
        self.edge_markers = t.get("edge_markers")
        self.neigh = t.get("neighbors")
        self.triangles = t.get("triangles", np.array([]))
        self.triangle_attributes = t.get("triangle_attributes", np.array([]))
        # self.iverts = t.get("triangles", np.array([]))
        self.segments = t.get("segments", np.array([]))
        self.segment_markers = t.get("segment_markers", np.array([]))

    @property
    def ncpl(self) -> int:
        return self.triangles.shape[0]

    @property
    def nvert(self) -> int:
        return self.vertices.shape[0]

    @property
    def verts(self):
        """Alias for retro-compatibility."""
        return self.vertices

    @property
    def iverts(self):
        """Alias for retro-compatibility."""
        return self.triangles

    def plot(
        self,
        ax=None,
        layer=0,
        edgecolor="k",
        facecolor="none",
        cmap="Dark2",
        a=None,
        masked_values=None,
        **kwargs,
    ):
        """
        Plot the grid.  This method will plot the grid using the shapefile
        that was created as part of the build method.

        Note that the layer option is not working yet.

        Parameters
        ----------
        ax : matplotlib.pyplot axis
            The plot axis.  If not provided it, plt.gca() will be used.
            If there is not a current axis then a new one will be created.
        layer : int
            Layer number to plot
        cmap : string
            Name of colormap to use for polygon shading (default is 'Dark2')
        edgecolor : string
            Color name.  (Default is 'scaled' to scale the edge colors.)
        facecolor : string
            Color name.  (Default is 'scaled' to scale the face colors.)
        a : numpy.ndarray
            Array to plot.
        masked_values : iterable of floats, ints
            Values to mask.
        kwargs : dictionary
            Keyword arguments that are passed to
            PatchCollection.set(``**kwargs``).  Some common kwargs would be
            'linewidths', 'linestyles', 'alpha', etc.

        Returns
        -------
        None

        """
        from ..discretization import VertexGrid
        from ..plot import PlotMapView

        cell2d = self.get_cell2d()
        vertices = self.get_vertices()
        ncpl = len(cell2d)

        modelgrid = VertexGrid(vertices=vertices, cell2d=cell2d, ncpl=ncpl, nlay=1)

        pmv = PlotMapView(modelgrid=modelgrid, ax=ax, layer=layer)
        if a is None:
            pc = pmv.plot_grid(facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        else:
            pc = pmv.plot_array(
                a, masked_values=masked_values, cmap=cmap, edgecolor=edgecolor, **kwargs
            )

        return pc

    def get_boundary_marker_array(self):
        """
        Get an integer array that has boundary markers

        Returns
        -------
        iedge : ndarray
            integer array of size ncpl containing a boundary ids.  The array
            contains zeros for cells that do not touch a boundary.  The
            boundary ids are the segment numbers for each segment in each
            polygon that is added with the add_polygon method.

        """
        iedge = np.zeros((self.ncpl), dtype=int)
        boundary_markers = np.unique(self.edge_markers)
        for ibm in boundary_markers:
            icells = self.get_edge_cells(ibm)
            iedge[icells] = ibm
        return iedge

    def plot_boundary(self, ibm, ax=None, **kwargs):
        """
        Plot a line and vertices for the specified boundary marker

        Parameters
        ----------
        ibm : integer
            plot the boundary for this boundary marker

        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        idx = np.asarray(self.edge_markers == ibm).nonzero()[0]
        for i in idx:
            iv1, iv2 = self.edges[i]
            x1 = self.node["x"][iv1]
            x2 = self.node["x"][iv2]
            y1 = self.node["y"][iv1]
            y2 = self.node["y"][iv2]
            ax.plot([x1, x2], [y1, y2], **kwargs)

    def plot_vertices(self, ax=None, **kwargs):
        """
        Plot the mesh vertices

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.node["x"], self.node["y"], lw=0, **kwargs)

    def label_vertices(self, ax=None, onebased=True, **kwargs):
        """
        Label the mesh vertices with their vertex numbers

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        onebased : bool
            Make the labels one-based if True so that they correspond to
            what would be written to MODFLOW.

        kwargs : dictionary
            dictionary of arguments to pass to ax.text()

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        for i in range(self.nvert):
            x = self.vertices[i, 0]
            y = self.vertices[i, 1]
            s = i
            if onebased:
                s += 1
            ax.text(x, y, str(s), **kwargs)

    def plot_centroids(self, ax=None, **kwargs):
        """
        Plot the cell centroids

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        kwargs : dictionary
            dictionary of arguments to pass to ax.plot()

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        xcyc = self.get_xcyc()
        ax.plot(xcyc[:, 0], xcyc[:, 1], lw=0, **kwargs)

    def label_cells(self, ax=None, onebased=True, **kwargs):
        """
        Label the cells with their cell numbers

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
           axis to add the plot to.  (default is plt.gca())

        onebased : bool
            Make the labels one-based if True so that they correspond to
            what would be written to MODFLOW.

        kwargs : dictionary
            dictionary of arguments to pass to ax.text()

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        xcyc = self.get_xcyc()
        for i in range(xcyc.shape[0]):
            x = xcyc[i, 0]
            y = xcyc[i, 1]
            s = i
            if onebased:
                s += 1
            ax.text(x, y, str(s), **kwargs)

    def get_xcyc(self):
        """
        Get a 2-dimensional array of x and y cell center coordinates.

        Returns
        -------
        xcyc : ndarray
            column 0 contains the x coordinates and column 1 contains the
            y coordinates

        """
        xcyc = np.empty((self.ncpl, 2), dtype=float)
        for i, icell2d in enumerate(self.triangles):
            points = []
            for iv in icell2d:
                x = self.vertices[iv, 0]
                y = self.vertices[iv, 1]
                points.append((x, y))
            xc, yc = centroid_of_polygon(points)
            xcyc[i, 0] = xc
            xcyc[i, 1] = yc
        return xcyc

    def get_cell2d(self):
        """
        Get a list of the information needed for the MODFLOW DISV Package.

        Returns
        -------
        cell2d : list (of lists)
            innermost list contains cell number, x, y, number of vertices, and
            then the vertex numbers comprising the cell.

        """
        cell2d = []
        xcyc = self.get_xcyc()
        for i, icell2d in enumerate(self.triangles):
            ic2dr = list(icell2d[::-1])
            cell2d.append([i, xcyc[i, 0], xcyc[i, 1], len(icell2d)] + ic2dr)
        return cell2d

    def get_vertices(self):
        """
        Get a list of vertices in the form needed for the MODFLOW DISV Package.

        Returns
        -------
        vertices : list (of lists)
            innermost list contains vertex number, x, and y

        """
        vertices = []
        for i, row in enumerate(self.verts):
            vertices.append([i, row[0], row[1]])
        return vertices

    def get_edge_cells(self, ibm):
        """
        Get a list of cell numbers that correspond to the specified boundary
        marker.

        Parameters
        ----------
        ibm : integer
            boundary marker value

        Returns
        -------
        cell_list : list
            list of zero-based cell numbers

        """
        # Create the edge dictionary if it doesn't exist
        if self.edgedict is None:
            self._create_edge_dict()

        # Create a list of cells for boundary marker ibm
        cell_list = []
        edgedict = self.edgedict
        for n, ivlist in enumerate(self.triangles):
            itmp = ivlist + [ivlist[0]]
            for i in range(len(ivlist)):
                ie = (itmp[i], itmp[i + 1])
                if ie in edgedict:
                    if edgedict[ie] == ibm:
                        cell_list.append(n)

        return cell_list

    def get_cell_edge_length(self, n, ibm):
        """
        Get the length of the edge for cell n that corresponds to
        boundary marker ibm

        Parameters
        ----------
        n : int
            cell number.  0 <= n < self.ncpl

        ibm : integer
            boundary marker number

        Returns
        -------
        length : float
            Length of the edge along that boundary marker.  Will
            return None if cell n does not touch boundary marker.

        """

        assert 0 <= n < self.ncpl, "Not a valid cell number"

        # Create the edge dictionary if it doesn't exist
        if self.edgedict is None:
            self._create_edge_dict()

        ivlist = self.triangles[n]
        itmp = ivlist + [ivlist[0]]
        d = None
        for i in range(len(ivlist)):
            iv1 = itmp[i]
            iv2 = itmp[i + 1]
            ie = (itmp[i], itmp[i + 1])
            if ie in self.edgedict:
                if self.edgedict[ie] == ibm:
                    x1, y1 = self.vertices[iv1]
                    x2, y2 = self.vertices[iv2]
                    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    return d
        return d

    def get_attribute_array(self):
        """
        Return an array containing the attribute value for each cell.  These
        are the attribute values that are passed into the add_region() method.

        Returns
        -------
        attribute_array : ndarray

        """
        return self.triangle_attributes

    def _initialize_vars(self):
        self.file_prefix = "_triangle"
        self._active_domain = None
        self._polygons = []
        self._holes = []
        self._regions = []
        self.vertices = np.array([])
        self.triangles = np.array([])
        self.edgedict = None

    def _get_triangulate_input(self) -> Dict[str, npt.NDArray[np.float64]]:
        nvert = 0
        for p in self._polygons:
            nvert += len(p)
        if self._nodes is not None:
            nvert += self._nodes.shape[0]
        nodes = []
        segs = []
        tot_index = 0
        for p in self._polygons:
            n_seg_in_poly = len(p)
            for vi, vertex in enumerate(p):
                nodes.append([vertex[0], vertex[1]])
                segs.append((vi + tot_index, (vi + 1) % n_seg_in_poly + tot_index))
            tot_index += n_seg_in_poly

        if self._nodes is not None:
            for i in range(self._nodes.shape[0]):
                nodes.append([self._nodes[i, 0], self._nodes[i, 1]])
        res = {"vertices": np.array(nodes), "segments": np.array(segs)}

        if len(self._holes) != 0:
            res["holes"] = np.array(self._holes)
        if len(self._regions) != 0:
            res["regions"] = self._get_regions()

        return res

    def _get_regions(self):
        return np.array([[*_[0], _[1], _[2]] for _ in self._regions])

    def _create_edge_dict(self):
        """
        Create the edge dictionary

        """
        edgedict = {}
        for iv1, iv2, iseg in (self.edges, self.edge_markers):
            if iseg != 0:
                edgedict[(iv1, iv2)] = iseg
                edgedict[(iv2, iv1)] = iseg
        self.edgedict = edgedict
