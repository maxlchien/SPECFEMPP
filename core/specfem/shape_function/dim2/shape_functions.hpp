#pragma once

#include <type_traits>
#include <vector>
namespace specfem::shape_function {

/** @brief Calculate shape functions for a 2D element given natural
 * coordinates \f$ (\xi, \gamma) \f$.
 *
 * Returns a vector containing the values of the shape functions \f$ N_a(\xi,
 * \gamma) \f$ at the specified integration point, where \f$ a = 1 \dots
 * \text{ngnod} \f$.
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ coordinate of the point.
 * @param gamma \f$ \gamma \f$ coordinate of the point.
 * @param ngnod Total number of control nodes per element.
 * @return std::vector<T> shape function values \f$ N_a(\xi, \gamma) \f$.
 */
template <typename T>
std::vector<T> shape_function(const T xi, const T gamma, const int ngnod);

/** @brief Calculate shape function derivatives for a 2D element given natural
 * coordinates \f$ (\xi, \gamma) \f$.
 *
 * Returns a matrix of derivatives:
 * \f[
 * \frac{\partial \mathbf{N}}{\partial \boldsymbol{\xi}} =
 * \begin{pmatrix}
 * \frac{\partial N_1}{\partial \xi} & \dots & \frac{\partial
 * N_{\text{ngnod}}}{\partial \xi} \\
 * \frac{\partial N_1}{\partial \gamma} & \dots & \frac{\partial
 * N_{\text{ngnod}}}{\partial \gamma}
 * \end{pmatrix}
 * \f]
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ coordinate of the point.
 * @param gamma \f$ \gamma \f$ coordinate of the point.
 * @param ngnod Total number of control nodes per element.
 * @return std::vector<std::vector<T>> Matrix of shape function derivatives
 * (size: \f$ 2 \times \text{ngnod} \f$).
 */
template <typename T>
std::vector<std::vector<T> >
shape_function_derivatives(const T xi, const T gamma, const int ngnod);

} // namespace specfem::shape_function
