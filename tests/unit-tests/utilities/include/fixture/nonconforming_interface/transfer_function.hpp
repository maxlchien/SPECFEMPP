#pragma once

#include "initializers.hpp"
#include "specfem_setup.hpp"

#include "Kokkos_Environment.hpp"
#include <array>
#include <tuple>
#include <type_traits>

namespace specfem::test::fixture {
template <typename Initializer> struct TransferFunction2D;

/**
 * @brief Test transfer function container.
 * @tparam Initializer Transfer function initialization strategy
 */
template <typename Initializer> struct TransferFunction2D {
  static_assert(
      std::is_base_of_v<TransferFunctionInitializer2D::Base, Initializer>,
      "TransferFunction2D needs an TransferFunctionInitializer2D!");

public:
  using TransferFunctionInitializer = Initializer;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static constexpr int num_edges = Initializer::num_edges;

  static constexpr int nquad_edge = Initializer::nquad_edge;

  static constexpr int nquad_intersection = Initializer::nquad_intersection;

  static std::string description() { return Initializer::description(); }

private:
  std::array<std::array<std::array<type_real, nquad_intersection>, nquad_edge>,
             num_edges>
      _transfer_function;
  using TransferFunctionView =
      Kokkos::View<type_real[num_edges][nquad_edge][nquad_intersection],
                   memory_space>;

public:
  /**
   * @brief Construct transfer function with initializer.
   * @param initializer Initialization strategy
   */
  TransferFunction2D(const Initializer &initializer) {

    for (int ielem = 0; ielem < num_edges; ++ielem) {
      for (int ipoint_edge = 0; ipoint_edge < nquad_edge; ++ipoint_edge) {
        for (int ipoint_intersection = 0;
             ipoint_intersection < nquad_intersection; ++ipoint_intersection) {
          (*this)(ielem, ipoint_edge, ipoint_intersection) = 0;
        }
      }
    }
    Initializer::init_transfer_function(*this);
  }

  /**
   * @brief Get Kokkos view of transfer function data.
   * @return Kokkos view for device access
   */
  TransferFunctionView get_view() const {
    TransferFunctionView view("transfer_function_view");
    for (size_t i = 0; i < num_edges; ++i) {
      for (size_t j = 0; j < nquad_edge; ++j) {
        for (size_t k = 0; k < nquad_intersection; ++k) {
          view(i, j, k) = (*this)(i, j, k);
        }
      }
    }
    return view;
  }

  /**
   * @brief Access transfer function values.
   * @param i Edge index
   * @param j Element quadrature index
   * @param k Intersection quadrature index
   * @return Reference to transfer function value
   */
  type_real &operator()(const int i, const int j, const int k) {
    return _transfer_function[i][j][k];
  }

  const type_real &operator()(const int i, const int j, const int k) const {
    return _transfer_function[i][j][k];
  }
};

namespace TransferFunctionInitializer2D {

// =================================================================================================
//      Transfer Function Initializers
// =================================================================================================

/** Zero transfer function initializer with ngll = nquad_intersection = 5 */
struct Zero : Base {
  static constexpr int num_edges = 1;
  static constexpr int nquad_edge = 5;
  static constexpr int nquad_intersection = 5;
  static void
  init_transfer_function(TransferFunction2D<Zero> &transfer_function) {
    // already zero-initialized don't need to do anything.
  }

  static std::string description() {
    return "A blank transfer function. This should zero out all values";
  }
};

template <typename EdgeQuadratureInitializer,
          typename IntersectionQuadratureInitializer>
struct FromQuadratureRules : Base {
  static constexpr int num_edges = 1;
  using IntersectionQuadrature =
      QuadratureRule<IntersectionQuadratureInitializer>;
  using EdgeQuadrature = QuadratureRule<EdgeQuadratureInitializer>;
  static constexpr int nquad_edge = EdgeQuadrature::nquad;
  static constexpr int nquad_intersection = IntersectionQuadrature::nquad;

  // static constexpr std::array<type_real, nquad_edge> edge_quadrature_points =
  //     EdgeQuadrature::quadrature_points;
  // static constexpr std::array<type_real, nquad_intersection>
  //     intersection_quadrature_points =
  //         IntersectionQuadrature::quadrature_points;

  static void init_transfer_function(
      TransferFunction2D<FromQuadratureRules> &transfer_function) {

    for (int ipoint_edge = 0; ipoint_edge < nquad_edge; ipoint_edge++) {
      for (int ipoint_intersection = 0;
           ipoint_intersection < nquad_intersection; ipoint_intersection++) {
        transfer_function(0, ipoint_edge, ipoint_intersection) =
            EdgeQuadrature::evaluate(
                ipoint_edge,
                IntersectionQuadrature::quadrature_points[ipoint_intersection]);
      }
    }
  }
  static std::string description() {
    return "GLL1 ({-1, 1} on the edge), GLL2 ({-1, 0, 1} on the intersection)";
  }
};

// =================================================================================================

} // namespace TransferFunctionInitializer2D

template <typename AnalyticalFieldInitializer,
          typename EdgeQuadratureInitializer,
          typename IntersectionQuadratureInitializer>
struct AnalyticalFieldTransfer2D {
  using AnalyticalField = AnalyticalField1D<AnalyticalFieldInitializer>;
  using IntersectionQuadrature =
      QuadratureRule<IntersectionQuadratureInitializer>;
  using EdgeQuadrature = QuadratureRule<EdgeQuadratureInitializer>;
  using EdgeFieldInitializer =
      EdgeFieldInitializer2D::FromAnalyticalField<AnalyticalFieldInitializer,
                                                  EdgeQuadratureInitializer>;
  using TransferFunctionInitializer =
      TransferFunctionInitializer2D::FromQuadratureRules<
          EdgeQuadratureInitializer, IntersectionQuadratureInitializer>;
  using TransferAndEdgeInitializerTuple =
      std::tuple<TransferFunctionInitializer, EdgeFieldInitializer>;

  static type_real evaluate_at_intersection_point(const int &iedge,
                                                  const int &iquad_intersection,
                                                  const int &icomp) {
    return AnalyticalField::evaluate(
        iedge, IntersectionQuadrature::quadrature_points[iquad_intersection],
        icomp);
  }
};

} // namespace specfem::test::fixture
