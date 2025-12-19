#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <array>
#include <type_traits>

namespace specfem::test::fixture {

template <typename Initializer> struct AnalyticalField1D {
  static_assert(
      std::is_base_of_v<AnalyticalFieldInitializer1D::Base, Initializer>,
      "AnalyticalField1D needs an AnalyticalFieldInitializer1D!");

  static constexpr int num_edges = Initializer::num_edges;
  static constexpr int num_components = Initializer::num_components;

  static std::string description() { return Initializer::description(); }
  static type_real evaluate(const int &iedge, const type_real &coord,
                            const int &icomp) {
    return Initializer::evaluate(iedge, coord, icomp);
  }
};

namespace AnalyticalFieldInitializer1D {

template <int power> struct xi_to_the : Base {
  static constexpr int power_taken = power;
  static constexpr int num_edges = 1;
  static constexpr int num_components = 1;
  static type_real evaluate(const int &iedge, const type_real &coord,
                            const int &icomp) {
    return std::pow(coord, power);
  }

  static std::string description() {
    return std::string("xi^") + std::to_string(power);
  }
};
} // namespace AnalyticalFieldInitializer1D

} // namespace specfem::test::fixture
