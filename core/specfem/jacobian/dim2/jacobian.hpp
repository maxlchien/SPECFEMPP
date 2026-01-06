#pragma once

#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem::jacobian {

/**
 * @brief Compute global locations \f$(x, z)\f$ from shape function matrix
 * calculated at \f$(\xi, \gamma)\f$.
 *
 * @param coorg Global control node locations \f$(x_a, z_a)\f$.
 * @param ngnod Total number of control nodes per element.
 * @param xi \f$\xi\f$ value of the point.
 * @param gamma \f$\gamma\f$ value of the point.
 * @return specfem::point::global_coordinates<specfem::dimension::type::dim2>
 *         The computed \f$(x, z)\f$ coordinates for the point.
 */
specfem::point::global_coordinates<specfem::dimension::type::dim2>
compute_locations(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute Jacobian matrix at \f$(\xi, \gamma)\f$.
 *
 * Calculates the partial derivatives and the determinant of the Jacobian.
 *
 * @param coorg View of coordinates required for the element.
 * @param ngnod Total number of control nodes per element.
 * @param xi \f$\xi\f$ value of the point.
 * @param gamma \f$\gamma\f$ value of the point.
 * @return specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
 * false> Structure containing partial derivatives
 *         \f$(\partial \xi / \partial x, \partial \gamma / \partial x,
 *             \partial \xi / \partial z, \partial \gamma / \partial z)\f$
 *         and the Jacobian determinant.
 */
specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
compute_jacobian(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real gamma);

/**
 * @brief Compute Jacobian matrix using precomputed shape function derivatives.
 *
 * @param coorg View of coordinates required for the element.
 * @param ngnod Total number of control nodes per element.
 * @param dershape2D Derivative of shape function matrix.
 * @return specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
 * false> Computed Jacobian matrix and determinant.
 */
specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
compute_jacobian(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const std::vector<std::vector<type_real> > &dershape2D);

} // namespace specfem::jacobian
