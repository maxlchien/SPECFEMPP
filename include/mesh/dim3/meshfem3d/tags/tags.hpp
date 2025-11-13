/**
 * @file tags.hpp
 * @brief MESHFEM3D element tagging system for 3D spectral element
 * classification
 *
 * This file implements element tagging for 3D spectral element meshes based on
 * MESHFEM3D material database information. It extracts material classification
 * from Materials containers and creates tags_container objects for each
 * element.
 *
 * Currently sets all boundary tags to `none` as other boundary conditions are
 * not yet implemented in the MESHFEM3D workflow.
 *
 * @see specfem::mesh::meshfem3d::Materials
 * @see specfem::mesh::impl::tags_container
 */
#pragma once

#include "enumerations/interface.hpp"
#include "mesh/dim3/meshfem3d/materials/materials.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::mesh::meshfem3d {

template <specfem::dimension::type DimensionTag> struct tags;

/**
 * @brief Element tagging system for 3D MESHFEM3D spectral elements
 *
 * Extracts material classification from MESHFEM3D Materials containers and
 * creates tags_container objects for each spectral element. Uses parallel
 * initialization to assign medium and property tags based on material data.
 *
 * Currently assigns boundary_tag::none to all elements since other boundary
 * conditions are not yet implemented in the MESHFEM3D workflow.
 *
 * @code
 * specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3>
 * materials;
 * // ... populate materials ...
 *
 * specfem::mesh::meshfem3d::tags<specfem::dimension::type::dim3> tags(
 *     nspec, materials);
 * @endcode
 *
 * @see specfem::mesh::meshfem3d::Materials::get_material_type
 * @see specfem::mesh::impl::tags_container
 */
template <> struct tags<specfem::dimension::type::dim3> {
  /**
   * @brief Dimension tag for compile-time type identification
   */
  static constexpr auto dimension_tag = specfem::dimension::type::dim3;

  /** @name Constructors */
  /** @{ */

  /**
   * @brief Default constructor creating empty tags container
   */
  tags() = default;

  /**
   * @brief Construct tags from MESHFEM3D materials database
   *
   * Extracts material classification (medium and property tags) from the
   * materials container for each element using parallel initialization.
   * Sets all boundary tags to `none` since other boundary conditions are
   * not yet implemented.
   *
   * @param nspec Total number of spectral elements in the mesh
   * @param materials MESHFEM3D Materials container with material data
   *
   * @see specfem::mesh::meshfem3d::Materials::get_material_type
   */
  tags(const int nspec,
       specfem::mesh::meshfem3d::Materials<dimension_tag> &materials);

  /** @} */

  /** @name Data Members */
  /** @{ */

  /** @brief Total number of spectral elements in the mesh */
  int nspec;

  /**
   * @brief Kokkos host view containing tags for all elements
   *
   * Each entry contains medium, property, and boundary tags for one spectral
   * element. Boundary tags are currently set to `none` for all elements.
   */
  Kokkos::View<specfem::mesh::impl::tags_container *,
               Kokkos::DefaultHostExecutionSpace>
      tags_container;

  /** @} */
};

} // namespace specfem::mesh::meshfem3d
