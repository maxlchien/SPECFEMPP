#pragma once

#include "accessor.hpp"
#include "container.hpp"

namespace specfem::data_access {

template <typename IndexType, typename ContainerType, typename AccessorType>
struct CheckCompatibility {
private:
  static constexpr bool check_dimension =
      IndexType::dimension_tag == ContainerType::dimension_tag &&
      IndexType::dimension_tag == AccessorType::dimension_tag;
  static constexpr bool check_data_class =
      ContainerType::data_class == AccessorType::data_class;

public:
  static constexpr bool value = check_dimension && check_data_class;

  static_assert(check_dimension, "Dimension tags do not match");
  static_assert(check_data_class, "Data classes do not match");
  static_assert(specfem::data_access::is_container<ContainerType>::value,
                "ContainerType is not a container");
  static_assert(AccessorType::accessor_type == IndexType::accessor_type,
                "AccessorType and IndexType have incompatible accessors");
};

template <typename T, typename = void>
struct is_jacobian_matrix : std::false_type {};

template <typename T>
struct is_jacobian_matrix<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::jacobian_matrix> >
    : std::true_type {};

template <typename T, typename = void>
struct is_field_derivatives : std::false_type {};

template <typename T>
struct is_field_derivatives<
    T,
    std::enable_if_t<T::data_class ==
                     specfem::data_access::DataClassType::field_derivatives> >
    : std::true_type {};

template <typename T, typename = void> struct is_source : std::false_type {};

template <typename T>
struct is_source<T,
                 std::enable_if_t<T::data_class ==
                                  specfem::data_access::DataClassType::source> >
    : std::true_type {};

template <typename T, typename = void> struct is_boundary : std::false_type {};

template <typename T>
struct is_boundary<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::boundary> >
    : std::true_type {};

template <typename T, typename = void>
struct is_properties : std::false_type {};

template <typename T>
struct is_properties<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::properties> >
    : std::true_type {};

template <typename T, typename = void> struct is_stress : std::false_type {};

template <typename T>
struct is_stress<T,
                 std::enable_if_t<T::data_class ==
                                  specfem::data_access::DataClassType::stress> >
    : std::true_type {};

template <typename T, typename = void> struct is_field : std::false_type {};

template <typename T>
struct is_field<
    T, std::enable_if_t<
           T::data_class == specfem::data_access::DataClassType::displacement ||
           T::data_class == specfem::data_access::DataClassType::velocity ||
           T::data_class == specfem::data_access::DataClassType::acceleration ||
           T::data_class == specfem::data_access::DataClassType::mass_matrix> >
    : std::true_type {};

template <typename T, typename = void>
struct is_displacement : std::false_type {};

template <typename T>
struct is_displacement<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::displacement> >
    : std::true_type {};

template <typename T, typename = void> struct is_velocity : std::false_type {};

template <typename T>
struct is_velocity<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::velocity> >
    : std::true_type {};

template <typename T, typename = void>
struct is_acceleration : std::false_type {};

template <typename T>
struct is_acceleration<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::acceleration> >
    : std::true_type {};

template <typename T, typename = void>
struct is_mass_matrix : std::false_type {};

template <typename T>
struct is_mass_matrix<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::mass_matrix> >
    : std::true_type {};

template <typename T, typename = void>
struct is_index_type : std::false_type {};

template <typename T>
struct is_index_type<
    T, std::enable_if_t<
           T::data_class == specfem::data_access::DataClassType::index ||
           T::data_class == specfem::data_access::DataClassType::mapped_index ||
           T::data_class == specfem::data_access::DataClassType::edge_index> >
    : std::true_type {};

template <typename T, typename = void>
struct is_assembly_index : std::false_type {};

template <typename T>
struct is_assembly_index<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::assembly_index> >
    : std::true_type {};

template <typename T, typename = void>
struct is_edge_index : std::false_type {};

template <typename T>
struct is_edge_index<
    T, std::enable_if_t<T::data_class ==
                        specfem::data_access::DataClassType::edge_index> >
    : std::true_type {};

template <typename T, typename = void>
struct is_coupled_interface : std::false_type {};

template <typename T>
struct is_coupled_interface<
    T,
    std::enable_if_t<T::data_class ==
                     specfem::data_access::DataClassType::coupled_interface> >
    : std::true_type {};

template <typename T, typename = void>
struct is_intersection_factor : std::false_type {};

template <typename T>
struct is_intersection_factor<
    T,
    std::enable_if_t<T::data_class ==
                     specfem::data_access::DataClassType::intersection_factor> >
    : std::true_type {};

template <typename T, typename = void>
struct is_intersection_normal : std::false_type {};

template <typename T>
struct is_intersection_normal<
    T,
    std::enable_if_t<T::data_class ==
                     specfem::data_access::DataClassType::intersection_normal> >
    : std::true_type {};

template <typename T, typename = void>
struct is_transfer_function_self : std::false_type {};

template <typename T>
struct is_transfer_function_self<
    T, std::enable_if_t<T::data_class == specfem::data_access::DataClassType::
                                             transfer_function_self> >
    : std::true_type {};

template <typename T, typename = void>
struct is_transfer_function_coupled : std::false_type {};

template <typename T>
struct is_transfer_function_coupled<
    T, std::enable_if_t<T::data_class == specfem::data_access::DataClassType::
                                             transfer_function_coupled> >
    : std::true_type {};

template <typename T, typename = void>
struct is_nonconforming_interface : std::false_type {};

template <typename T>
struct is_nonconforming_interface<
    T, std::enable_if_t<T::data_class == specfem::data_access::DataClassType::
                                             nonconforming_interface> >
    : std::true_type {};

template <typename T, typename = void>
struct is_conforming_interface : std::false_type {};

template <typename T>
struct is_conforming_interface<
    T, std::enable_if_t<T::data_class == specfem::data_access::DataClassType::
                                             conforming_interface> >
    : std::true_type {};

template <typename T, typename = void>
struct is_packed_accessor : std::false_type {};

template <typename T>
struct is_packed_accessor<T, std::enable_if_t<T::n_accessors >= 1> >
    : std::true_type {};

} // namespace specfem::data_access
