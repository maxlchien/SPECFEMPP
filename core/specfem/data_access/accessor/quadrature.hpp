#pragma once

#include "datatypes/interface.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access/accessor.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::data_access {

template <specfem::data_access::DataClassType DataClass,
          specfem::dimension::type DimensionTag, bool UseSIMD>
struct Accessor<specfem::data_access::AccessorType::quadrature, DataClass,
                DimensionTag, UseSIMD> {
  constexpr static auto accessor_type =
      specfem::data_access::AccessorType::quadrature;
  constexpr static auto data_class = DataClass;
  constexpr static auto dimension_tag = DimensionTag;
};

template <typename T, typename = void>
struct is_quadrature : std::false_type {};

template <typename T>
struct is_quadrature<
    T, std::enable_if_t<T::accessor_type ==
                        specfem::data_access::AccessorType::quadrature> >
    : std::true_type {};

} // namespace specfem::data_access
