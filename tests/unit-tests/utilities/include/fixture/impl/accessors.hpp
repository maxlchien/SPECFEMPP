#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/medium.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/data_access/data_class.hpp"
namespace specfem::test_fixture::impl {

/**
 * @brief Baseline view for a nonconforming data accessor
 * (core/specfem/chunk_edge/nonconforming_interface.hpp)
 *
 * @tparam InterfaceTag
 * @tparam BoundaryTag
 * @tparam DataClassType
 * @tparam Axes The size of the view along each axis.
 */
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::data_access::DataClassType DataClassType, int... Axes>
struct NonconformingAccessorPatch2D
    : specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge, DataClassType,
          specfem::dimension::type::dim2, false /* UseSIMD */> {
public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;
  /// View type for storing intersection scaling factors
  using DataViewType =
      Kokkos::View<type_real[Axes]...,
                   Kokkos::DefaultExecutionSpace::memory_space>;

private:
  /// Underlying view storing transfer function matrix data
  DataViewType data_;

public:
  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPatch2D() = default;

  NonconformingAccessorPatch2D(const std::string &name) : data_(name) {}

  /**
   * @brief Access transfer function matrix element
   * @tparam Indices Index types for multi-dimensional access
   * @param indices Element indices (edge, intersection_quad, edge_quad)
   * @return Reference to matrix element
   */
  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) const {
    return data_(indices...);
  }
};

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadElement, int NQuadIntersection>
struct NonconformingTransferFunctionSelfPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::transfer_function_self,
          NumberElements, NQuadElement, NQuadIntersection> {
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_element = NQuadElement;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadElement, int NQuadIntersection>
struct NonconformingTransferFunctionCoupledPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::transfer_function_coupled,
          NumberElements, NQuadElement, NQuadIntersection> {
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_element = NQuadElement;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection>
struct NonconformingIntersectionNormalPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::intersection_normal,
          NumberElements, NQuadIntersection, 2> {
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
};
template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag, int NumberElements,
          int NQuadIntersection>
struct NonconformingIntersectionFactorPatch
    : NonconformingAccessorPatch2D<
          InterfaceTag, BoundaryTag,
          specfem::data_access::DataClassType::intersection_factor,
          NumberElements, NQuadIntersection> {
  static constexpr int chunk_size = NumberElements;
  static constexpr int n_quad_intersection = NQuadIntersection;
};

} // namespace specfem::test_fixture::impl
