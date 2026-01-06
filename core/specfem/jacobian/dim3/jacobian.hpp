#pragma once

#include "kokkos_abstractions.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem::jacobian {

/**
 * @brief Compute global locations \f$(x, y, z)\f$ from shape function matrix
 * calculated at \f$(\xi, \eta, \gamma)\f$.
 *
 * @param coorg Global control node locations.
 * @param ngnod Total number of control nodes per element.
 * @param xi \f$\xi\f$ value of the point.
 * @param eta \f$\eta\f$ value of the point.
 * @param gamma \f$\gamma\f$ value of the point.
 * @return specfem::point::global_coordinates<specfem::dimension::type::dim3>
 *         The computed \f$(x, y, z)\f$ coordinates.
 */
specfem::point::global_coordinates<specfem::dimension::type::dim3>
compute_locations(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

/**
 * @brief Compute Jacobian matrix at \f$(\xi, \eta, \gamma)\f$.
 *
 * Calculates the partial derivatives and the determinant of the Jacobian.
 *
 * @param coorg View of coordinates required for the element.
 * @param ngnod Total number of control nodes per element.
 * @param xi \f$\xi\f$ value of the point.
 * @param eta \f$\eta\f$ value of the point.
 * @param gamma \f$\gamma\f$ value of the point.
 * @return specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true,
 * false> Structure containing partial derivatives and the Jacobian determinant.
 */
specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
compute_jacobian(
    const Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma);

/**
 * @brief Compute Jacobian matrix for 3D elements using generic views.
 *
 * This function calculates the Jacobian matrix and its determinant given nodal
 * coordinates and shape function derivatives.
 *
 * @tparam CoordinateView Type of the coordinate view (e.g., Kokkos::View).
 * @tparam ShapeDerivativesView Type of the shape function derivatives view.
 * @param coordinates View of nodal coordinates.
 * @param shape_derivatives View of shape function derivatives.
 * @return specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true,
 * false> Computed Jacobian matrix and determinant.
 */
template <typename CoordinateView, typename ShapeDerivativesView>
KOKKOS_FUNCTION
    specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
    compute_jacobian(const CoordinateView coordinates,
                     const ShapeDerivativesView shape_derivatives) {
  const int ngnod = coordinates.extent(0);
  type_real xxi = 0.0, yxi = 0.0, zxi = 0.0;
  type_real xeta = 0.0, yeta = 0.0, zeta = 0.0;
  type_real xgamma = 0.0, ygamma = 0.0, zgamma = 0.0;
  for (int in = 0; in < ngnod; in++) {
    xxi += shape_derivatives(0, in) * coordinates(in, 0);
    yxi += shape_derivatives(0, in) * coordinates(in, 1);
    zxi += shape_derivatives(0, in) * coordinates(in, 2);
    xeta += shape_derivatives(1, in) * coordinates(in, 0);
    yeta += shape_derivatives(1, in) * coordinates(in, 1);
    zeta += shape_derivatives(1, in) * coordinates(in, 2);
    xgamma += shape_derivatives(2, in) * coordinates(in, 0);
    ygamma += shape_derivatives(2, in) * coordinates(in, 1);
    zgamma += shape_derivatives(2, in) * coordinates(in, 2);
  }
  auto jacobian = xxi * (yeta * zgamma - ygamma * zeta) -
                  xeta * (yxi * zgamma - ygamma * zxi) +
                  xgamma * (yxi * zeta - yeta * zxi);
  const type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
  const type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
  const type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
  const type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
  const type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
  const type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
  const type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
  const type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
  const type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;
  return { xix, xiy, xiz, etax, etay, etaz, gammax, gammay, gammaz, jacobian };
}

} // namespace specfem::jacobian
