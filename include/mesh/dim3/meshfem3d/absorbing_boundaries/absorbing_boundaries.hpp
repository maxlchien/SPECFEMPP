/**
 * @file absorbing_boundaries.hpp
 * @brief Absorbing boundary conditions management for 3D spectral element
 * meshes
 *
 * This file contains the AbsorbingBoundaries template specialization for 3D
 * meshes. The class manages absorbing boundary face information read from
 * MESHFEM3D database files, which implement Stacey absorbing boundary
 * conditions to prevent spurious wave reflections at domain boundaries.
 */

#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh::meshfem3d {

/**
 * @brief Template structure for managing absorbing boundaries in spectral
 * element meshes
 *
 * @tparam DimensionTag Dimension type tag for template specialization
 */
template <specfem::dimension::type DimensionTag> struct AbsorbingBoundaries;

/**
 * @brief Absorbing boundary conditions manager for 3D spectral element meshes
 *
 * This template specialization manages absorbing boundary face information
 * for 3D spectral element simulations. The class stores data read from
 * MESHFEM3D database files. These conditions prevent spurious wave reflections
 * by absorbing outgoing waves at the domain boundaries.
 *
 * @code
 * // Example usage with MESHFEM3D database file
 * std::ifstream database_stream("MESH/absorbing_boundaries.bin",
 * std::ios::binary); auto control_nodes = read_control_nodes(database_stream);
 *
 * // Read absorbing boundaries from database
 * auto absorbing_boundaries =
 *     specfem::io::mesh_impl::fortran::dim3::meshfem3d::read_absorbing_boundaries(
 *         database_stream, control_nodes, mpi);
 *
 * // Access boundary face information
 * for (int iface = 0; iface < absorbing_boundaries.nfaces; ++iface) {
 *     int element_index = absorbing_boundaries.index_mapping(iface);
 *     auto face_type = absorbing_boundaries.face_type(iface);
 *     // Process absorbing boundary face...
 * }
 * @endcode
 */
template <> struct AbsorbingBoundaries<specfem::dimension::type::dim3> {

private:
  /** @brief Kokkos view type for storing element indices on host memory */
  using IndexViewType = Kokkos::View<int *, Kokkos::HostSpace>;

  /** @brief Kokkos view type for storing face types on host memory */
  using FaceViewType =
      Kokkos::View<specfem::mesh_entity::dim3::type *, Kokkos::HostSpace>;

public:
  /** @brief Dimension tag for template specialization (always dim3) */
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  /**
   * @brief Default constructor
   *
   * Creates an empty absorbing boundaries object with zero faces.
   * Use this constructor when absorbing boundaries are not required
   * or will be populated later.
   */
  AbsorbingBoundaries() = default;

  /**
   * @brief Parameterized constructor for absorbing boundaries
   *
   * Constructs an absorbing boundaries object with the specified number
   * of boundary faces. Allocates Kokkos host memory views for storing
   * element indices and face types read from MESHFEM3D database files.
   *
   * @param nfaces Number of absorbing boundary faces to allocate storage for
   *
   * @note The constructor initializes Kokkos views with labeled memory for
   * debugging and profiling purposes.
   *
   * @code
   * // Create absorbing boundaries for 120 faces
   * specfem::mesh::meshfem3d::AbsorbingBoundaries<specfem::dimension::type::dim3>
   *     boundaries(120);
   *
   * // Views are now allocated and ready for data from MESHFEM3D
   * assert(boundaries.nfaces == 120);
   * assert(boundaries.index_mapping.extent(0) == 120);
   * assert(boundaries.face_type.extent(0) == 120);
   * @endcode
   */
  AbsorbingBoundaries(const int nfaces)
      : nfaces(nfaces),
        index_mapping("specfem::mesh::AbsorbingBoundaries::index_mapping",
                      nfaces),
        face_type("specfem::mesh::AbsorbingBoundaries::face_type", nfaces) {}

  /** @brief Total number of absorbing boundary faces */
  int nfaces;

  /**
   * @brief Mapping from boundary face index to spectral element index
   *
   * This Kokkos host view stores the spectral element indices that contain
   * absorbing boundary faces.
   */
  IndexViewType index_mapping;

  /**
   * @brief Type identifier for each absorbing boundary face
   *
   * This Kokkos host view stores the face type (bottom, top, front, back,
   * left, right) for each absorbing boundary face.
   */
  FaceViewType face_type;
};

} // namespace specfem::mesh::meshfem3d
