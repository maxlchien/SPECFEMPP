#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d {

/**
 * @brief Reads control nodes from a meshfem3D binary database file.
 *
 * This function reads control node data from a meshfem3D-generated binary file
 * containing node coordinates for 3D spectral element mesh generation. The
 * function follows the standard fortran binary format used by SPECFEM3D
 * meshfem3D module.
 *
 * The file format consists of:
 * 1. Number of control nodes (integer)
 * 2. For each node: node_index, x_coordinate, y_coordinate, z_coordinate
 *
 * All data is read using `specfem::io::read_fortran_line` which handles the
 * standard fortran unformatted binary record structure with record length
 * markers before and after each data block.
 *
 * @param stream Input file stream positioned at the control nodes section
 * @param mpi MPI communication object for distributed processing context
 * @return ControlNodes object containing node count and 3D coordinate data
 *
 * @throws std::runtime_error if file reading fails or data format is invalid
 *
 * @code
 * // Typical usage in meshfem3D file reading:
 * std::ifstream stream("mesh_database.bin", std::ios::binary);
 * // const auto* mpi = initialize_MPI_context();
 *
 * auto control_nodes =
 * specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_control_nodes(stream,
 * mpi);
 *
 * // Access the data
 * int total_nodes = control_nodes.nnodes;
 * auto x = control_nodes.coordinates(node_id, 0);  // x-coordinate
 * auto y = control_nodes.coordinates(node_id, 1);  // y-coordinate
 * auto z = control_nodes.coordinates(node_id, 2);  // z-coordinate
 * @endcode
 *
 * @see specfem::mesh::meshfem3d::ControlNodes
 * @see specfem::io::read_fortran_line
 * @see specfem::io::read_3d_mesh (main mesh reading function)
 */
specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
read_control_nodes(std::ifstream &stream, const specfem::MPI::MPI *mpi);

} // namespace specfem::io::mesh::impl::fortran::dim3::meshfem3d
