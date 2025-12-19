#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include <string>

namespace specfem::test::fixture {

template <typename Initializer> struct QuadratureRule {
  static_assert(std::is_base_of_v<QuadratureRuleInitializer::Base, Initializer>,
                "QuadratureRule needs an QuadratureRuleInitializer!");

public:
  static constexpr int nquad =
      sizeof(decltype(Initializer::quadrature_points)) /
      sizeof(typename Initializer::value_type);
  static constexpr std::array<typename Initializer::value_type, nquad>
      quadrature_points = Initializer::quadrature_points;

  template <typename FloatingType>
  static constexpr FloatingType evaluate(const int &iquad,
                                         const FloatingType &x) {
    FloatingType val = 1;
    for (int i = 0; i < nquad; i++) {
      if (i != iquad) {
        val *= (x - quadrature_points[i]) /
               (quadrature_points[iquad] - quadrature_points[i]);
      }
    }
    return val;
  }
};

namespace QuadratureRuleInitializer {

struct GLL1 : Base {
  using value_type = type_real;
  static constexpr std::array<type_real, 2> quadrature_points = { -1, 1 };

  static std::string description() { return "GLL1 (-1, 1)"; }
};
struct GLL2 : Base {
  using value_type = type_real;
  static constexpr std::array<type_real, 3> quadrature_points = { -1, 0, 1 };
  static std::string description() { return "GLL2 (-1, 0, 1)"; }
};

struct Asymm5Point : Base {
  using value_type = type_real;
  static constexpr std::array<type_real, 5> quadrature_points = { -1, -0.8,
                                                                  -0.5, 0.2,
                                                                  0.7 };
  static std::string description() {
    return "5 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};
struct Asymm4Point : Base {
  using value_type = type_real;
  static constexpr std::array<type_real, 4> quadrature_points = { -0.3, 0, 0.4,
                                                                  0.6 };
  static std::string description() {
    return "4 point asymmetric (low exactness interpolating quadrature for "
           "testing)";
  }
};

} // namespace QuadratureRuleInitializer

} // namespace specfem::test::fixture
