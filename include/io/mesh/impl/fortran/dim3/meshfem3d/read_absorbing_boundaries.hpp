#pragma once

#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"
#include <fstream>

namespace specfem::io::mesh_impl::fortran::dim3::meshfem3d {
/**
 * @brief Read absorbing boundary face information from MESHFEM3D database files
 *
 * Reads binary Fortran-formatted absorbing boundary data and constructs an
 * AbsorbingBoundaries object containing face indices and face types(bottom,
 * top, front, back, left, right).
 *
 * @param stream Input file stream positioned at absorbing boundary data
 * section, opened in binary mode
 * @param control_nodes ControlNodes object containing node coordinates for face
 * matching
 * @param mpi MPI communication interface for parallel processing
 *
 * @return AbsorbingBoundaries object with face count, element indices, and face
 * types
 *
 * @throws std::runtime_error If database format is invalid or face matching
 * fails
 */
specfem::mesh::meshfem3d::AbsorbingBoundaries<specfem::dimension::type::dim3>
read_absorbing_boundaries(
    std::ifstream &stream,
    const specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
        &control_nodes,
    const specfem::MPI::MPI *mpi);

} // namespace specfem::io::mesh_impl::fortran::dim3::meshfem3d
