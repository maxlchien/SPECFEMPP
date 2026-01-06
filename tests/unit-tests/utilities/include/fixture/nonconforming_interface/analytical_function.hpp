#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

namespace specfem::test_fixture {

namespace AnalyticalFunctionType {

/**
 * @brief Describes a function f(x) = x^k for a power k
 *
 * @tparam power the exponent.
 */
template <int power> struct Power : AnalyticalFunctionType {
  static constexpr int num_components = 1;
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    return { (type_real)std::pow(coord, power) };
  }

  static std::string description() {
    return std::string("xi^") + std::to_string(power);
  }
};

/**
 * @brief Concatenates k functions into an array-valued function.
 *
 */
template <typename... AnalyticalFunctions>
struct Vectorized : AnalyticalFunctionType {
  static_assert(
      ((std::is_base_of_v<AnalyticalFunctionType, AnalyticalFunctions>) && ...),
      "Vectorized needs all of its parameters to be of "
      "AnalyticalFunctionType!");
  static constexpr int num_components =
      ((AnalyticalFunctions::num_components) + ...);
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    std::array<type_real, num_components> arr;
    auto it = arr.begin();
    (
        [&]() {
          const auto sub = AnalyticalFunctions::evaluate(coord);
          std::copy(sub.begin(), sub.end(), it);
          it += AnalyticalFunctions::num_components;
        }(),
        ...);
    return arr;
  }

  static std::string description() {
    return std::string("Vectorized(") +
           ((AnalyticalFunctions::description() + ",") + ...) + ")";
  }
};

/**
 * @brief Sums k functions.
 *
 */
template <typename... AnalyticalFunctions> struct Sum : AnalyticalFunctionType {
  static_assert(
      ((std::is_base_of_v<AnalyticalFunctionType, AnalyticalFunctions>) && ...),
      "Sum needs all of its parameters to be of "
      "AnalyticalFunctionType!");
  static constexpr int num_components =
      std::min((AnalyticalFunctions::num_components)...);

  static_assert(
      ((AnalyticalFunctions::num_components == num_components) && ...),
      "Sum needs all of its parameters to have the same number of components!");
  static std::array<type_real, num_components>
  evaluate(const type_real &coord) {
    std::array<type_real, num_components> arr{ 0 };
    (
        [&]() {
          const auto sub = AnalyticalFunctions::evaluate(coord);
          for (int icomp = 0; icomp < num_components; ++icomp) {
            arr[icomp] += sub[icomp];
          }
        }(),
        ...);
    return arr;
  }

  static std::string description() {
    return std::string("Sum(") +
           ((AnalyticalFunctions::description() + ",") + ...) + ")";
  }
};

} // namespace AnalyticalFunctionType

} // namespace specfem::test_fixture
