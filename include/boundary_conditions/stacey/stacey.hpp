#pragma once

#include "enumerations/boundary.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

/**
 * @file stacey.hpp
 * @brief Stacey absorbing boundary conditions
 *
 */

namespace specfem {
namespace boundary_conditions {

using stacey_type =
    std::integral_constant<specfem::element::boundary_tag,
                           specfem::element::boundary_tag::stacey>;

/**
 * @brief Apply Stacey absorbing boundary conditions
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointFieldType Point field type
 * @tparam PointAccelerationType Point acceleration type
 * @param boundary Boundary object
 * @param property Property object
 * @param field Field object
 * @param acceleration Acceleration object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointFieldType, typename PointAccelerationType>
KOKKOS_FUNCTION void impl_apply_boundary_conditions(
    const stacey_type &, const PointBoundaryType &boundary,
    const PointPropertyType &property, const PointFieldType &field,
    PointAccelerationType &acceleration);

/**
 * @brief Compute mass matrix terms for Stacey absorbing boundary conditions
 *
 * @tparam PointBoundaryType Point boundary type
 * @tparam PointPropertyType Point property type
 * @tparam PointMassMatrixType Point mass matrix type
 * @param dt Time step
 * @param boundary Boundary object
 * @param property Property object
 * @param mass_matrix Mass matrix object
 */
template <typename PointBoundaryType, typename PointPropertyType,
          typename PointMassMatrixType>
KOKKOS_FUNCTION void impl_compute_mass_matrix_terms(
    const stacey_type &, const type_real dt, const PointBoundaryType &boundary,
    const PointPropertyType &property, PointMassMatrixType &mass_matrix);

} // namespace boundary_conditions
} // namespace specfem
