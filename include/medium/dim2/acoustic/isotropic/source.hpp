#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::acoustic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {

  using PointAccelerationType =
      specfem::point::field<PointPropertiesType::dimension_tag,
                            PointPropertiesType::medium_tag, false, false, true,
                            false, PointPropertiesType::simd::using_simd>;

  PointAccelerationType result;

  result.acceleration(0) = point_source.stf(0) *
                           point_source.lagrange_interpolant(0) /
                           point_properties.kappa();

  return result;
}

} // namespace medium
} // namespace specfem
