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


def update_cells(
    cells_def_arr: npt.NDArray[int],
    idx: int,
    bot_face_pts_id: Sequence[int],
    top_face_pts_id: Sequence[int],
) -> int:

    # Number of points composing the bot and top faces (it should be the same)
    n_pts_2d = len(top_face_pts_id)

    # Add NItems
    cells_def_arr[idx] = 1 + (n_pts_2d + 1) * 2 + n_pts_2d * 5

    # Add the number of faces
    idx += 1  # update idx
    cells_def_arr[idx] = 2 + n_pts_2d  # number of faces

    # Add top face
    idx += 1  # update idx
    cells_def_arr[idx] = n_pts_2d
    cells_def_arr[idx + 1 : idx + 1 + n_pts_2d] = top_face_pts_id
    # update idx
    idx += n_pts_2d

    # Add side faces
    for n in range(n_pts_2d - 1):
        cells_def_arr[idx + 1] = 4
        cells_def_arr[idx + 2] = top_face_pts_id[n]
        cells_def_arr[idx + 3] = top_face_pts_id[n + 1]
        cells_def_arr[idx + 4] = bot_face_pts_id[n + 1]
        cells_def_arr[idx + 5] = bot_face_pts_id[n]
        idx += 5

    # last face
    cells_def_arr[idx + 1] = 4
    cells_def_arr[idx + 2] = top_face_pts_id[n_pts_2d - 1]
    cells_def_arr[idx + 3] = top_face_pts_id[0]
    cells_def_arr[idx + 4] = bot_face_pts_id[0]
    cells_def_arr[idx + 5] = bot_face_pts_id[n_pts_2d - 1]
    idx += 5

    # Add bottom face
    # Add top face
    idx += 1
    cells_def_arr[idx] = n_pts_2d
    cells_def_arr[idx + 1 : idx + 1 + n_pts_2d] = bot_face_pts_id
    # update idx
    idx += n_pts_2d + 1

    return idx


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

    _update_cells = update_cells
    if is_use_numba:
        import_optional_dependency(
            "numba",
            error_message="pv_extrude with `is_use_numba` True requires numba installed.",
        )
        import numba

        _update_cells = numba.njit(update_cells, cache=True)

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

    nlayers: int = np.size(thickness)
    points = np.tile(surfmesh.points, (nlayers + 1, 1))
    nc = surfmesh.number_of_points

    # Definition of the z-coordinates for each layer
    for i in range(nlayers):
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

    # pre-allocate the arrays => more efficient than list and compatible with numba
    # for large grids with millions of cells.
    cells = np.zeros(ncells * nlayers, dtype=np.int64)
    celltypes = np.zeros(surfmesh.number_of_cells * nlayers, dtype=np.uint8)
    layers = np.zeros(surfmesh.number_of_cells * nlayers, dtype=np.uint8)

    ind = 0
    ncell = 0
    cda_idx = 0
    for i in range(surfmesh.number_of_cells):
        npoints = surfmesh.cells[ind]
        edges = surfmesh.cells[ind + 1 : ind + 1 + npoints]

        # iterate the layers
        for j in range(nlayers):
            # update the type of cell
            # celltypes[ncell] = ctype.setdefault(npoints, vtk.VTK_POLYHEDRON)
            celltypes[ncell] = ctype.setdefault(npoints, vtk.VTK_POLYHEDRON)
            # celltype between 3 and 6
            if celltypes[ncell] != vtk.VTK_POLYHEDRON:
                # update number of points
                cells[cda_idx] = 2 * npoints
                # update lower side
                cells[cda_idx + 1 : cda_idx + 1 + npoints] = edges + j * nc
                # update upper side
                cells[cda_idx + 1 + npoints : cda_idx + 1 + 2 * npoints] = (
                    edges + (j + 1) * nc
                )
                # update the cells comptor
                cda_idx += 1 + 2 * npoints
            # celltype not covered by "classic" objects => use of polyhedron
            # which is more tricky to define because each face must be
            # defined individually.
            else:
                cda_idx = _update_cells(
                    cells, cda_idx, edges + j * nc, edges + (j + 1) * nc
                )
            volume.append(area[i] * thickness[j])

            # update the layer for the current cell and then the number of cells
            layers[ncell] = j
            ncell += 1

        # update comptor
        ind += 1 + npoints

    mesh = pv.UnstructuredGrid(cells, celltypes, points)

    mesh["Volume"] = volume
    mesh["Layer"] = layers

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

        # At this point cc has shape (n_cell in 2d, 3, len(thickness))
        total_th: float = 0.0
        for i, th in enumerate(thickness):
            # the first index is treated after => otherwise it modies cc[:, 2, 0] inplace
            if i == 0:
                total_th += th
                continue
            # update z
            # print(total_th + th / 2.0)
            cc[:, 2, i] += cc[:, 2, 0] + total_th + th / 2.0
            total_th += th
        # handle index 0
        cc[:, 2, 0] += thickness[0] / 2.0

        # Add the voronoi cell centers
        mesh.cell_data["Cell centers"] = cc.transpose(0, 2, 1).reshape(
            ncells * len(thickness), 3
        )

    return mesh
