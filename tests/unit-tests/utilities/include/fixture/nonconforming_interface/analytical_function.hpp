#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <array>
#include <type_traits>

namespace specfem::test::fixture {

namespace AnalyticalFunctionType1D {

template <int power> struct Power : AnalyticalFunctionType1D {
  static constexpr int num_components = 1;
  static type_real evaluate(const type_real &coord) {
    return std::pow(coord, power);
  }

  static std::string description() {
    return std::string("xi^") + std::to_string(power);
  }
};
} // namespace AnalyticalFunctionType1D

} // namespace specfem::test::fixture
