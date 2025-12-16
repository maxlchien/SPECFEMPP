#pragma once
#include <stdexcept>
#include <type_traits>

#include "specfem_setup.hpp"

static constexpr char blank_description[] = "(No description.)";

namespace specfem::test::fixture {

template <typename Initializer> struct TransferFunction2D;

namespace impl {
template <typename T, typename = void> struct _has_num_edges : std::false_type {
  static constexpr int num_edges = -1;
};
template <typename T>
struct _has_num_edges<T, std::void_t<decltype(T::num_edges)> >
    : std::true_type {
  static constexpr int num_edges = T::num_edges;
};

template <typename T, typename = void>
struct _has_nquad_edge : std::false_type {
  static constexpr int nquad_edge = -1;
};
template <typename T>
struct _has_nquad_edge<T, std::void_t<decltype(T::nquad_edge)> >
    : std::true_type {
  static constexpr int nquad_edge = T::nquad_edge;
};

template <typename T, typename = void>
struct _has_nquad_intersection : std::false_type {
  static constexpr int nquad_intersection = -1;
};
template <typename T>
struct _has_nquad_intersection<T, std::void_t<decltype(T::nquad_intersection)> >
    : std::true_type {
  static constexpr int nquad_intersection = T::nquad_intersection;
};

template <typename T, typename = void>
struct _has_init_transfer_function : std::false_type {
  static constexpr void call(TransferFunction2D<T> &transfer_function) {
    throw std::runtime_error(
        "Called impl::_has_init_transfer_function::call(), "
        "which "
        "should be never called.");
  }
};
template <typename T>
struct _has_init_transfer_function<
    T, std::void_t<decltype(T::init_transfer_function)> > : std::true_type {
  static constexpr void call(TransferFunction2D<T> &transfer_function) {
    T::init_transfer_function(transfer_function);
  }
};

template <typename T, typename = void>
struct _has_edge_quadrature_points : std::false_type {
  static constexpr int nquad = 0;
  static constexpr std::array<type_real, 0> edge_quadrature_points = {};
};
template <typename T>
struct _has_edge_quadrature_points<
    T, std::void_t<decltype(T::edge_quadrature_points)> > : std::true_type {
  static constexpr int nquad = sizeof(decltype(T::edge_quadrature_points)) /
                               sizeof(decltype(T::edge_quadrature_points[0]));
  static constexpr std::array<type_real, nquad> edge_quadrature_points =
      T::edge_quadrature_points;
};

template <typename T, typename = void>
struct _has_intersection_quadrature_points : std::false_type {
  static constexpr int nquad = 0;
  static constexpr std::array<type_real, 0> intersection_quadrature_points = {};
};
template <typename T>
struct _has_intersection_quadrature_points<
    T, std::void_t<decltype(T::intersection_quadrature_points)> >
    : std::true_type {
  static constexpr int nquad =
      sizeof(decltype(T::intersection_quadrature_points)) /
      sizeof(decltype(T::intersection_quadrature_points[0]));
  static constexpr std::array<type_real, nquad> intersection_quadrature_points =
      T::intersection_quadrature_points;
};

template <typename T, typename = void>
struct _has_description : std::false_type {
  static std::string description() { return std::string(blank_description); }
};
template <typename T>
struct _has_description<T, std::void_t<decltype(T::description)> >
    : std::true_type {
  static std::string description() { return T::description(); }
};

} // namespace impl
} // namespace specfem::test::fixture
