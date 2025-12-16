#pragma once

#include "initializer_value_check.hpp"

#include "Kokkos_Environment.hpp"

namespace specfem::test::fixture {

/**
 * @brief Test field container.
 * @tparam Initializer Field initialization strategy
 */
template <typename Initializer> struct EdgeFunction2D {
public:
  using FunctionInitializer = Initializer;
  static constexpr int num_edges = Initializer::num_edges;
  static constexpr int num_components = Initializer::num_components;
  static constexpr int nquad_edge = Initializer::nquad_edge;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static std::string description() {
    return impl::_has_description<Initializer>::description();
  }

private:
  std::array<std::array<std::array<type_real, num_components>, nquad_edge>,
             num_edges>
      _field;
  using FieldView =
      Kokkos::View<type_real[num_edges][nquad_edge][num_components],
                   memory_space>;

public:
  /**
   * @brief Construct field with initializer.
   * @param initializer Initialization strategy
   */
  EdgeFunction2D(const FunctionInitializer &initializer) {
    for (int ielem = 0; ielem < num_edges; ++ielem) {
      for (int ipoint_edge = 0; ipoint_edge < nquad_edge; ++ipoint_edge) {
        for (int icomp = 0; icomp < num_components; ++icomp) {
          (*this)(ielem, ipoint_edge, icomp) = 0;
        }
      }
    }
    FunctionInitializer::init_function(*this);
  }

  /**
   * @brief Get Kokkos view of field data.
   * @return Kokkos view for device access
   */
  FieldView get_view() const {
    FieldView view("field_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          view(i, j, k) = (*this)(i, j, k);
        }
      }
    }
    return view;
  }

  /**
   * @brief Access field values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Component index
   * @return Reference to field value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _field[i][j][k];
  }
  const type_real &operator()(const int i, const int j, const int k) const {
    return _field[i][j][k];
  }
};

/**
 * @brief
 *
 */
namespace EdgeFunctionInitializer2D {
struct Uniform {
  static constexpr int num_edges = 1;
  static constexpr int num_components = 1;
  static constexpr int nquad_edge = 5;
  static void init_function(EdgeFunction2D<Uniform> &edge_function) {
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          edge_function(i, j, k) = 1;
        }
      }
    }
  }
};

template <typename FieldInitializer, typename EdgePointsInitializer>
struct FromEdgeField {
  using FieldInitializerType = FieldInitializer;

  using EdgePointsInitializer_has_edge_quadrature_points =
      impl::_has_edge_quadrature_points<EdgePointsInitializer>;

  static_assert(
      EdgePointsInitializer_has_edge_quadrature_points::value,
      "EdgePointsInitializer must have field `edge_quadrature_points`");

  static constexpr int num_edges = FieldInitializer::num_edges;
  static constexpr int num_components = FieldInitializer::num_components;
  static constexpr int nquad_edge =
      EdgePointsInitializer_has_edge_quadrature_points::nquad;
  static constexpr std::array<type_real, nquad_edge> edge_quadrature_points =
      EdgePointsInitializer_has_edge_quadrature_points::edge_quadrature_points;

  static std::string description() {
    return std::string("EdgeField-initialized function (description: \"") +
           impl::_has_description<FieldInitializer>::description() + "\")";
  }

  static void init_function(
      EdgeFunction2D<FromEdgeField<FieldInitializer, EdgePointsInitializer> >
          &edge_function) {
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < num_components; ++k) {
          edge_function(i, j, k) =
              FieldInitializer::evaluate(i, edge_quadrature_points[j], k);
        }
      }
    }
  }
};

} // namespace EdgeFunctionInitializer2D
} // namespace specfem::test::fixture
