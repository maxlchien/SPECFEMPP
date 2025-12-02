#pragma once

#include "specfem/data_access.hpp"

namespace specfem::point {

namespace impl {

template <int NQuadIntersection, specfem::dimension::type DimensionTag,
          specfem::data_access::DataClassType DataClass,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct nonconforming_transfer_function
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point, DataClass, DimensionTag,
          false> {
public:
  static constexpr auto interface_tag = InterfaceTag;
  static constexpr auto boundary_tag = BoundaryTag;
  static constexpr auto n_quad_intersection = NQuadIntersection;
  static constexpr auto connection_tag =
      specfem::connections::type::nonconforming;

  using TransferViewType =
      specfem::datatype::VectorPointViewType<type_real, NQuadIntersection,
                                             false>;

  TransferViewType transfer_function;

  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function(const TransferViewType &transfer_function)
      : transfer_function(transfer_function) {}

  KOKKOS_INLINE_FUNCTION
  nonconforming_transfer_function() = default;

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION const auto &operator()(Indices... indices) const {
    return transfer_function(indices...);
  }

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION auto &operator()(Indices... indices) {
    return transfer_function(indices...);
  }
};

} // namespace impl

template <int NQuadIntersection, specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_self = impl::nonconforming_transfer_function<
    NQuadIntersection, DimensionTag,
    specfem::data_access::DataClassType::transfer_function_self, InterfaceTag,
    BoundaryTag>;

template <int NQuadIntersection, specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_coupled = impl::nonconforming_transfer_function<
    NQuadIntersection, DimensionTag,
    specfem::data_access::DataClassType::transfer_function_coupled,
    InterfaceTag, BoundaryTag>;

template <typename... Accessors>
struct NonconformingAccessorPack
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::nonconforming_interface,
          specfem::dimension::type::dim2, false>,
      public Accessors... {

private:
  using accessor_base = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::nonconforming_interface,
      specfem::dimension::type::dim2, false>;

public:
  constexpr static auto interface_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::interface_tag;
  constexpr static auto boundary_tag =
      std::tuple_element_t<0, std::tuple<Accessors...> >::boundary_tag;
  constexpr static size_t n_accessors = sizeof...(Accessors);
  using packed_accessors = std::tuple<Accessors...>;
  constexpr static auto connection_tag =
      specfem::connections::type::nonconforming;

  constexpr static auto dimension_tag = accessor_base::dimension_tag;
  constexpr static auto data_class = accessor_base::data_class;
  constexpr static auto accessor_type = accessor_base::accessor_type;
  constexpr static bool using_simd = accessor_base::using_simd;

  static_assert(
      (std::is_same_v<
           std::integral_constant<specfem::dimension::type,
                                  Accessors::dimension_tag>,
           std::integral_constant<specfem::dimension::type, dimension_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "dimension_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::interface::interface_tag,
                                             Accessors::interface_tag>,
                      std::integral_constant<specfem::interface::interface_tag,
                                             interface_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "interface_tag");

  static_assert(
      (std::is_same_v<std::integral_constant<specfem::element::boundary_tag,
                                             Accessors::boundary_tag>,
                      std::integral_constant<specfem::element::boundary_tag,
                                             boundary_tag> > &&
       ...),
      "All Accessors in NonconformingAccessorPack must have the same "
      "boundary_tag");

  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack() = default;

  KOKKOS_INLINE_FUNCTION
  NonconformingAccessorPack(const Accessors &...accessors)
      : Accessors(accessors)... {};

  template <typename... Indices>
  KOKKOS_INLINE_FUNCTION type_real operator()(Indices... indices) const =
      delete;
};

template <int NQuadIntersection, specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
using transfer_function_pack =
    NonconformingAccessorPack<transfer_function_coupled<
        NQuadIntersection, DimensionTag, InterfaceTag, BoundaryTag> >;

} // namespace specfem::point
