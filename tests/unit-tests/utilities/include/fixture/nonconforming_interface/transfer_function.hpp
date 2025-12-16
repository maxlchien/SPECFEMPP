#pragma once

#include "initializer_value_check.hpp"

#include "Kokkos_Environment.hpp"
#include <array>
#include <type_traits>

namespace specfem::test::fixture {

namespace TransferFunctionInitializer2D {

namespace impl {

/**
 * @brief Evaluates the Lagrange interpolation polynomials at a given point x:
 *     $$L_j(x) = \prod_{k \ne j} \frac{x - \xi_k}{\xi_j - \xi_k}$$
 *
 * @tparam QPArrayType - datatype of the quadrature points array
 * @param quadrature_points - the array of $\xi_k$
 * @param nquad - the number of quadrature points
 * @param poly_index - the index of the lagrange polynomial to evaluate
 * @param x - the point to evaluate at
 * @return type_real - the evaluated $L_j(x)$, where $j$ is given by
 * `poly_index`
 */
template <typename QPArrayType>
KOKKOS_FORCEINLINE_FUNCTION type_real
eval_lagrange(const QPArrayType &quadrature_points, const int &nquad,
              const int &poly_index, const type_real &x) {
  type_real val = 1;
  for (int i = 0; i < nquad; i++) {
    if (i != poly_index) {
      val *= (x - quadrature_points[i]) /
             (quadrature_points[poly_index] - quadrature_points[i]);
    }
  }
  return val;
}

/**
 * @brief Initialize a standard transfer function by the knot values.
 *
 * @param transfer_function transfer function to populate
 * @param element_index the index to populate (0 <= element_index < chunk_size)
 * @param edge_quadrature_points a list of quadrature points along the edge
 * @param intersection_quadrature_points a list of quadrature points at the
 * intersection (in edge coordinates)
 */
template <typename Initializer, typename EdgeKnotArray,
          typename IntersectionKnotArray>
void initialize_transfer_function_values(
    TransferFunction2D<Initializer> &transfer_function,
    const int &element_index, const EdgeKnotArray &edge_quadrature_points,
    const IntersectionKnotArray &intersection_quadrature_points) {

  for (int ipoint_edge = 0; ipoint_edge < transfer_function.nquad_edge;
       ipoint_edge++) {
    for (int ipoint_intersection = 0;
         ipoint_intersection < transfer_function.nquad_intersection;
         ipoint_intersection++) {
      transfer_function(element_index, ipoint_edge, ipoint_intersection) =
          eval_lagrange(edge_quadrature_points, transfer_function.nquad_edge,
                        ipoint_edge,
                        intersection_quadrature_points[ipoint_intersection]);
    }
  }
}
// overload to support initializer lists
template <typename Initializer, std::size_t nquad_edge,
          std::size_t nquad_intersection>
void initialize_transfer_function_values(
    TransferFunction2D<Initializer> &transfer_function,
    const int &element_index,
    const type_real (&edge_quadrature_points)[nquad_edge],
    const type_real (&intersection_quadrature_points)[nquad_intersection]) {
  std::array<type_real, nquad_edge> edge_arr;

  for (int ipoint_edge = 0; ipoint_edge < transfer_function.nquad_edge;
       ipoint_edge++) {
    edge_arr[ipoint_edge] = edge_quadrature_points[ipoint_edge];
  }
  initialize_transfer_function_values(transfer_function, element_index,
                                      edge_arr, intersection_quadrature_points);
}

/**
 * @brief Recovers nquad_edge from either the field value or size of
 * edge_quadrature_points
 *
 * @tparam T
 */
template <typename T> struct _initval_nquad_edge {
  using has_nquad_edge = specfem::test::fixture::impl::_has_nquad_edge<T>;
  using has_edge_quadrature_points =
      specfem::test::fixture::impl::_has_edge_quadrature_points<T>;

  static_assert(
      has_nquad_edge::value || has_edge_quadrature_points::value,
      "Initializer::nquad_edge or "
      "Initializer::edge_quadrature_points must be defined, but neither are.");

  static_assert(
      (!(has_nquad_edge::value && has_edge_quadrature_points::value)) ||
          (has_nquad_edge::value && has_edge_quadrature_points::value &&
           has_nquad_edge::nquad_edge == has_edge_quadrature_points::nquad),
      "Initializer::nquad_edge != len(Initializer::edge_quadrature_points)");
  static constexpr int value = has_nquad_edge::value
                                   ? has_nquad_edge::nquad_edge
                                   : has_edge_quadrature_points::nquad;
};

/**
 * @brief Recovers nquad_intersection from either the field value or size of
 * intersection_quadrature_points
 *
 * @tparam T
 */
template <typename T> struct _initval_nquad_intersection {
  using has_nquad_intersection =
      specfem::test::fixture::impl::_has_nquad_intersection<T>;
  using has_intersection_quadrature_points =
      specfem::test::fixture::impl::_has_intersection_quadrature_points<T>;

  static_assert(has_nquad_intersection::value ||
                    has_intersection_quadrature_points::value,
                "Initializer::nquad_intersection or "
                "Initializer::intersection_quadrature_points must be defined, "
                "but neither are.");

  static_assert((!(has_nquad_intersection::value &&
                   has_intersection_quadrature_points::value)) ||
                    (has_nquad_intersection::value &&
                     has_intersection_quadrature_points::value &&
                     has_nquad_intersection::nquad_intersection ==
                         has_intersection_quadrature_points::nquad),
                "Initializer::nquad_intersection != "
                "len(Initializer::intersection_quadrature_points)");
  static constexpr int value = has_nquad_intersection::value
                                   ? has_nquad_intersection::nquad_intersection
                                   : has_intersection_quadrature_points::nquad;
};

/**
 * @brief If edge and intersection quadrature points are given, we implicitly
 * assume this call:
 *
 * @tparam T
 */
template <typename T, typename = void> struct _init_transfer_call {
  static constexpr bool has_implicit_call = false;

  static constexpr void call(TransferFunction2D<T> &transfer_function) {
    T::init_transfer_function(transfer_function);
  }
};
template <typename T>
struct _init_transfer_call<
    T,
    std::enable_if_t<
        specfem::test::fixture::impl::_has_edge_quadrature_points<T>::value &&
            specfem::test::fixture::impl::_has_intersection_quadrature_points<
                T>::value,
        void> > {
  static constexpr bool has_implicit_call = true;

  static_assert(
      !specfem::test::fixture::impl::_has_init_transfer_function<T>::value,
      "If specifying edge and intersection quadrature points, implicit "
      "`init_transfer_function` is assumed and used. Please remove explicit "
      "call.");

  static constexpr void call(TransferFunction2D<T> &transfer_function) {
    for (int i = 0; i < transfer_function.num_edges; i++) {
      initialize_transfer_function_values(transfer_function, i,
                                          T::edge_quadrature_points,
                                          T::intersection_quadrature_points);
    }
  }
};

} // namespace impl
} // namespace TransferFunctionInitializer2D

/**
 * @brief Test transfer function container.
 * @tparam Initializer Transfer function initialization strategy
 */
template <typename Initializer> struct TransferFunction2D {
public:
  using TransferFunctionInitializer = Initializer;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  static constexpr int num_edges = Initializer::num_edges;
  static_assert(impl::_has_num_edges<Initializer>::value,
                "Initializer must have static member `num_edges`");

  static constexpr int nquad_edge =
      TransferFunctionInitializer2D::impl::_initval_nquad_edge<
          Initializer>::value;
  static_assert(nquad_edge > 0, "Cannot recover `nquad_edge` from Initializer");

  static constexpr int nquad_intersection =
      TransferFunctionInitializer2D::impl::_initval_nquad_intersection<
          Initializer>::value;
  static_assert(nquad_intersection > 0,
                "Cannot recover `nquad_intersection` from Initializer");

  using init_transfer_function_call =
      TransferFunctionInitializer2D::impl::_init_transfer_call<Initializer>;

  static_assert(
      impl::_has_init_transfer_function<Initializer>::value ||
          init_transfer_function_call::has_implicit_call,
      "Either init_transfer_function must be provided by Initializer, "
      "or edge and intersection quadrature points must be specified.");

  static std::string description() {
    return impl::_has_description<Initializer>::description();
  }

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
    init_transfer_function_call::call(*this);
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
struct Zero {
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

/** GLL1 on edge, GLL2 on intersection */
struct GLL1_to_GLL2 {
  static constexpr int num_edges = 1;
  static constexpr int nquad_edge = 2;
  static constexpr int nquad_intersection = 3;
  static constexpr std::array<type_real, nquad_edge> edge_quadrature_points = {
    -1, 1
  };
  static constexpr std::array<type_real, nquad_intersection>
      intersection_quadrature_points = { -1, 0, 1 };

  static std::string description() {
    return "GLL1 ({-1, 1} on the edge), GLL2 ({-1, 0, 1} on the intersection)";
  }
};

/** Two non-gll quadrature rules (len(edge) > len(intersection)) */
struct ASYM5POINT_to_ASYM4POINT {
  static constexpr int num_edges = 1;
  static constexpr std::array<type_real, 5> edge_quadrature_points = {
    -1, -0.8, -0.5, 0.2, 0.7
  };
  static constexpr std::array<type_real, 4> intersection_quadrature_points = {
    -0.3, 0, 0.4, 0.6
  };

  static std::string description() {
    return "Transfer function from two asymmetric quadrature rules.";
  }
};

// =================================================================================================

} // namespace TransferFunctionInitializer2D

} // namespace specfem::test::fixture
