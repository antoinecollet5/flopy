from typing import Iterable, List, Sequence

import numpy as np
import numpy.typing as npt
import pyvista as pv

from .utl_import import import_optional_dependency


def create_unstructured_mesh(
    vertices: npt.NDArray[float], connectivities: List[List[int]]
):

    import_optional_dependency(
        "pyvista",
        error_message="VoronoiGrid.to_pyvista requires pyvista.",
    )
    import vtk

    """
    Creates a VTK unstructured mesh from the given points and connectivities.
    Only polygon cells are supported.
    """
    # Create an empty VTK unstructured grid
    mesh = vtk.vtkUnstructuredGrid()

    # Create a VTK points object from the input points
    vtk_points = vtk.vtkPoints()
    for vert in vertices:
        vtk_points.InsertNextPoint(vert)
    mesh.SetPoints(vtk_points)

    # Create a VTK cell array from the input connectivities
    vtk_cells = vtk.vtkCellArray()
    for connectivity in connectivities:
        n_points = len(connectivity)
        # print(n_points)
        vtk_cell = vtk.vtkPolygon()
        vtk_cell.GetPointIds().SetNumberOfIds(n_points)
        for j, point_index in enumerate(connectivity):
            vtk_cell.GetPointIds().SetId(j, point_index)
        vtk_cells.InsertNextCell(vtk_cell)

    mesh.SetCells(vtk.VTK_POLYGON, vtk_cells)

    return mesh


def add_dimension(points, dim="z"):

    z = np.zeros((np.shape(points)[0], 1))

    if dim.upper() == "Z" or dim.upper() == "XY":
        return np.hstack((points, z))
    elif dim.upper() == "X" or dim.upper() == "YZ":
        return np.hstack((z, points))
    elif dim.upper() == "Y" or dim.upper() == "XZ":
        return np.hstack((points[:, 0], z, points[:, 1]))
    else:
        raise ValueError("Parameter dim = " + str(dim) + " not implemented.")


def get_ncell4poly(npoints: int) -> int:
    if npoints < 3:
        raise RuntimeError(
            "The number of exterior points for the polygon should be at least 3."
        )
    if npoints == 3:  # triangle (2D) ==>  WEDGE (3D)
        return 7  #
    if npoints == 4:  # HEXAHEDRON
        return 9
    if npoints == 5:  # PENTAGONAL_PRISM
        return 11
    if npoints == 6:  # HEXAGONAL_PRISM
        return 13
    # POLYHEDRON
    return 2 + (npoints + 1) * 2 + npoints * 5


def get_polyhedron_connectivity(
    bot_face_pts_id: Sequence[int], top_face_pts_id: Sequence[int]
) -> List[int]:

    n_pts_2d = len(top_face_pts_id)
    # Add the number of faces
    polyhedron_connectivity = [2 + n_pts_2d]  # number of faces
    # Add top face
    polyhedron_connectivity += [n_pts_2d] + list(top_face_pts_id)
    # Add side faces
    for n in range(n_pts_2d - 1):
        polyhedron_connectivity += [4] + [
            top_face_pts_id[n],
            top_face_pts_id[n + 1],
            bot_face_pts_id[n + 1],
            bot_face_pts_id[n],
        ]
    # last face
    polyhedron_connectivity += [4] + [
        top_face_pts_id[n_pts_2d - 1],
        top_face_pts_id[0],
        bot_face_pts_id[0],
        bot_face_pts_id[n_pts_2d - 1],
    ]

    # Add bottom face
    polyhedron_connectivity += [len(bot_face_pts_id)] + list(bot_face_pts_id)
    # Add NItems
    polyhedron_connectivity.insert(0, len(polyhedron_connectivity))

    return polyhedron_connectivity


def pv_extrude(
    surfmesh: pv.UnstructuredGrid,
    thickness: Iterable[float],
    is_use_numba: bool = False,
) -> pv.UnstructuredGrid:
    """
    Use a topographic surface to create a 3D terrain-following mesh.

    Parameters
    ----------
    surfmesh : Pyvista.UnstructuredGrid with 2D-Elements
        Surface mesh.
    thickness : Iterable of float
        Thickness of layers in z-direction.

    Returns
    -------
    mesh : Pyvista.UnstructuredGrid with 3D-Elements
        Extruded surface mesh.

    """
    import_optional_dependency(
        "pyvista",
        error_message="VoronoiGrid.to_pyvista requires pyvista.",
    )
    import vtk

    surfmesh = surfmesh.compute_cell_sizes(length=False, area=True, volume=False)
    area = surfmesh["Area"]
    volume = list()

    # Typ des Geometry bzws. des Prismas
    ctype = {
        3: vtk.VTK_WEDGE,
        4: vtk.VTK_HEXAHEDRON,
        5: vtk.VTK_PENTAGONAL_PRISM,
        6: vtk.VTK_HEXAGONAL_PRISM,
        7: vtk.VTK_POLYHEDRON,
    }

    nlayer = np.size(thickness)
    points = np.tile(surfmesh.points, (nlayer + 1, 1))
    nc = surfmesh.number_of_points

    # Definition of the z-coordinates for each layer
    for i in range(nlayer):
        points[((i + 1) * nc) : (i + 2) * nc, 2] = (
            points[i * nc : ((i + 1) * nc), 2] + thickness[i]
        )

    # Define the total number of elements that will be in the cell definition
    ncells = 0
    ind = 0
    for i in range(surfmesh.number_of_cells):
        npoints = surfmesh.cells[ind]
        ncells += get_ncell4poly(surfmesh.cells[ind])
        ind = ind + 1 + npoints

    # create the cell vector
    cells_def_arr = np.zeros(ncells * nlayer, dtype=np.int64)

    cells = list()
    celltypes = list()
    layer = list()

    ind = 0
    for i in range(surfmesh.number_of_cells):
        npoints = surfmesh.cells[ind]
        edges = surfmesh.cells[ind + 1 : ind + 1 + npoints]
        ind = ind + 1 + npoints

        # _vtk.VTK_POLYHEDRON
        for j in range(nlayer):
            celltypes.append(ctype.setdefault(npoints, vtk.VTK_POLYHEDRON))

            # celltype between 3 and 6
            if celltypes[-1] != vtk.VTK_POLYHEDRON:
                cells.append(2 * npoints)  # number of points
                cells.extend(edges + j * nc)  # lower side
                cells.extend(edges + (j + 1) * nc)  # upper side
            # celltype not covered by "classic" objects => use of polyhedron
            # which is more tricky to define because each face must be
            # defined individually.
            else:
                cells += get_polyhedron_connectivity(
                    edges + j * nc, edges + (j + 1) * nc
                )

            volume.append(area[i] * thickness[j])
            layer.append(j)

    assert cells_def_arr.size == len(cells)

    mesh = pv.UnstructuredGrid(cells, np.array(celltypes), points)

    mesh["Volume"] = volume
    mesh["Layer"] = layer

    # add voronoi points
    if "Cell centers" in surfmesh.cell_data.keys():
        ncells = surfmesh.number_of_cells
        # duplicate for each layer
        # cc = np.tile(surfmesh.cell_data["Cell centers"], (len(thickness), 1))
        cc = np.repeat(
            surfmesh.cell_data["Cell centers"].reshape(ncells, -1, 1),
            repeats=len(thickness),
            axis=2,
        )
        prev_th = 0.0
        for i, th in enumerate(thickness):
            # update z
            cc[:, 2, i] = cc[:, 2, max(i, 0)] + prev_th / 2.0 + th / 2.0
            prev_th = th

            # Add the voronoi cell centers
            mesh.cell_data.set_array(
                cc.transpose(0, 2, 1).reshape(ncells * len(thickness), 3),
                "Cell centers",
            )

    return mesh
