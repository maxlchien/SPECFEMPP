#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

using PointJacobianMatrix2D =
    specfem::point::jacobian_matrix<specfem::dimension::type::dim2, false,
                                    false>;
using JacobianViewType2D = Kokkos::View<PointJacobianMatrix2D **,
                                        Kokkos::LayoutRight, Kokkos::HostSpace>;

/**
 * @brief Compute source array for a 2D tensor source using spatial derivatives.
 *
 * For moment tensor sources, computes source contributions by transforming
 * tensor components via spatial derivatives of Lagrange basis functions:
 * @f$ S_{i,jz,jx} = M_{i,0} \frac{\partial L}{\partial x} + M_{i,1}
 * \frac{\partial L}{\partial z} @f$
 *
 * Derivatives are computed using the chain rule with Jacobian matrix elements.
 *
 * @param source Tensor source containing moment tensor components
 * @param mesh Mesh providing quadrature information
 * @param jacobian_matrix Coordinate transformation Jacobians at GLL points
 * @param source_array Output array of shape (ncomponents, ngllz, ngllx)
 */
void from_tensor(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &source,
    const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh,
    const specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
        &jacobian_matrix,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

/**
 * @brief Helper function computing tensor source array with precomputed
 * Jacobians.
 *
 * Separates Jacobian extraction from computation for testability.
 *
 * @param tensor_source Tensor source object
 * @param element_jacobian_matrix Precomputed Jacobian matrices for the element
 * @param quadrature Quadrature containing GLL points
 * @param source_array Output array
 */
void compute_source_array_from_tensor_and_element_jacobian(
    const specfem::sources::tensor_source<specfem::dimension::type::dim2>
        &tensor_source,
    const JacobianViewType2D &element_jacobian_matrix,
    const specfem::assembly::mesh_impl::quadrature<
        specfem::dimension::type::dim2> &quadrature,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
