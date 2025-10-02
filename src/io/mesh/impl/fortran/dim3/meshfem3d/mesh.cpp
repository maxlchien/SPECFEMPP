#include "mesh/mesh.hpp"
#include "io/fortranio/interface.hpp"
#include "io/interface.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_absorbing_boundaries.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_control_nodes.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_materials.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_mpi_interfaces.hpp"
#include "io/mesh/impl/fortran/dim3/meshfem3d/read_pml_boundaries.hpp"
#include <fstream>
#include <stdexcept>
#include <string>

specfem::mesh::meshfem3d::mesh<specfem::dimension::type::dim3>
specfem::io::meshfem3d::read_3d_mesh(const std::string &mesh_parameters_file,
                                     const std::string &mesh_databases_file,
                                     const specfem::MPI::MPI *mpi) {
  // Read mesh parameters
  std::ifstream param_stream(mesh_parameters_file,
                             std::ios::in | std::ios::binary);
  if (!param_stream.is_open()) {
    throw std::runtime_error("Could not open mesh parameters file: " +
                             mesh_parameters_file);
  }

  bool mesh_of_earth_chunk;
  specfem::io::fortran_read_line(param_stream, &mesh_of_earth_chunk);

  // TODO (Rohit: EARTH_CHUNK_MESH): Add support for mesh of earth chunk
  if (mesh_of_earth_chunk) {
    throw std::runtime_error(
        "Mesh of earth chunk is not supported in SPECFEM++ yet.");
  }

  specfem::mesh::meshfem3d::mesh<specfem::dimension::type::dim3> mesh;

  mesh.control_nodes =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_control_nodes(
          param_stream, mpi);
  const auto [control_node_index, materials] =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_materials(
          param_stream, mesh.control_nodes.ngnod, mpi);
  mesh.control_nodes.control_node_index = control_node_index;
  mesh.materials = materials;
  mesh.absorbing_boundaries =
      specfem::io::mesh::impl::fortran::dim3::meshfem3d::
          read_absorbing_boundaries(param_stream, mesh.control_nodes, mpi);

  // CPML boundaries are not supported yet
  // TODO (Rohit: PML_BOUNDARIES): Add support for PML boundaries
  specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_pml_boundaries(
      param_stream, mpi);

  // MPI interfaces are not supported yet
  // TODO (Rohit: MPI_INTERFACES): Add support for MPI interfaces
  specfem::io::mesh::impl::fortran::dim3::meshfem3d::read_mpi_interfaces(
      param_stream, mpi);

  param_stream.close();

  return mesh;
}
