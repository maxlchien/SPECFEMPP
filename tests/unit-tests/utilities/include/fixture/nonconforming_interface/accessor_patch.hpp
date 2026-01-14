#pragma once

#include "../impl/accessors.hpp"
#include "Kokkos_Macros.hpp"
#include "initializers.hpp"
#include <string>

namespace specfem::test_fixture {
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer>
struct NonconformingAccessorPatch2D<
    InterfaceTag, BoundaryTag, Initializer,
    specfem::data_access::DataClassType::transfer_function_self>
    : specfem::test_fixture::impl::NonconformingTransferFunctionSelfPatch<
          InterfaceTag, BoundaryTag, Initializer::num_edges,
          Initializer::nquad_edge, Initializer::nquad_intersection> {
  static_assert(
      std::is_base_of_v<
          TransferFunctionInitializer2D::TransferFunctionInitializer2D,
          Initializer> ||
          std::is_base_of_v<
              IntersectionDataInitializer2D::IntersectionDataInitializer2D,
              Initializer>,
      "NonconformingAccessorPatch2D<...,transfer_function_self> needs an "
      "TransferFunctionInitializer2D or "
      "IntersectionDataInitializer2D!");
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPatch2D() = default;
  NonconformingAccessorPatch2D(const std::string &name)
      : specfem::test_fixture::impl::NonconformingTransferFunctionSelfPatch<
            InterfaceTag, BoundaryTag, Initializer::num_edges,
            Initializer::nquad_edge, Initializer::nquad_intersection>(
            name + " -- transfer_function_self") {
    const auto &arr = [&]() {
      if constexpr (std::is_base_of_v<TransferFunctionInitializer2D::
                                          TransferFunctionInitializer2D,
                                      Initializer>) {
        return Initializer::init_transfer_function();
      } else {
        return Initializer::init_transfer_function_self();
      }
    }();
    for (size_t i = 0; i < Initializer::num_edges; ++i) {
      for (size_t j = 0; j < Initializer::nquad_edge; ++j) {
        for (size_t k = 0; k < Initializer::nquad_intersection; ++k) {
          (*this)(i, j, k) = arr[i][j][k];
        }
      }
    }
  }
};
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer>
struct NonconformingAccessorPatch2D<
    InterfaceTag, BoundaryTag, Initializer,
    specfem::data_access::DataClassType::transfer_function_coupled>
    : specfem::test_fixture::impl::NonconformingTransferFunctionCoupledPatch<
          InterfaceTag, BoundaryTag, Initializer::num_edges,
          Initializer::nquad_edge, Initializer::nquad_intersection> {
  static_assert(
      std::is_base_of_v<
          TransferFunctionInitializer2D::TransferFunctionInitializer2D,
          Initializer> ||
          std::is_base_of_v<
              IntersectionDataInitializer2D::IntersectionDataInitializer2D,
              Initializer>,
      "NonconformingAccessorPatch2D<...,transfer_function_coupled> needs an "
      "TransferFunctionInitializer2D or "
      "IntersectionDataInitializer2D!");
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPatch2D() = default;
  NonconformingAccessorPatch2D(const std::string &name)
      : specfem::test_fixture::impl::NonconformingTransferFunctionCoupledPatch<
            InterfaceTag, BoundaryTag, Initializer::num_edges,
            Initializer::nquad_edge, Initializer::nquad_intersection>(
            name + " -- transfer_function_coupled") {
    const auto &arr = [&]() {
      if constexpr (std::is_base_of_v<TransferFunctionInitializer2D::
                                          TransferFunctionInitializer2D,
                                      Initializer>) {
        return Initializer::init_transfer_function();
      } else {
        return Initializer::init_transfer_function_coupled();
      }
    }();
    for (size_t i = 0; i < Initializer::num_edges; ++i) {
      for (size_t j = 0; j < Initializer::nquad_edge; ++j) {
        for (size_t k = 0; k < Initializer::nquad_intersection; ++k) {
          (*this)(i, j, k) = arr[i][j][k];
        }
      }
    }
  }
};
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, typename Initializer>
struct NonconformingAccessorPatch2D<
    InterfaceTag, BoundaryTag, Initializer,
    specfem::data_access::DataClassType::intersection_normal>
    : specfem::test_fixture::impl::NonconformingIntersectionNormalPatch<
          InterfaceTag, BoundaryTag, Initializer::num_edges,
          Initializer::nquad_intersection> {
  static_assert(
      std::is_base_of_v<
          IntersectionFunctionInitializer2D::IntersectionFunctionInitializer2D,
          Initializer> ||
          std::is_base_of_v<
              IntersectionDataInitializer2D::IntersectionDataInitializer2D,
              Initializer>,
      "NonconformingAccessorPatch2D<...,intersection_normal> needs an "
      "IntersectionFunctionInitializer2D or "
      "IntersectionDataInitializer2D!");
  KOKKOS_INLINE_FUNCTION NonconformingAccessorPatch2D() = default;
  NonconformingAccessorPatch2D(const std::string &name)
      : specfem::test_fixture::impl::NonconformingIntersectionNormalPatch<
            InterfaceTag, BoundaryTag, Initializer::num_edges,
            Initializer::nquad_intersection>(name + " -- intersection_normal") {
    const auto &arr = [&]() {
      if constexpr (std::is_base_of_v<TransferFunctionInitializer2D::
                                          TransferFunctionInitializer2D,
                                      Initializer>) {
        return Initializer::init_function();
      } else {
        return Initializer::init_intersection_normal();
      }
    }();
    for (size_t i = 0; i < Initializer::num_edges; ++i) {
      for (size_t k = 0; k < Initializer::nquad_intersection; ++k) {
        (*this)(i, k) = arr[i][k];
      }
    }
  }
};
} // namespace specfem::test_fixture
