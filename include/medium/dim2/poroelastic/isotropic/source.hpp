#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::medium {

template <typename PointSourceType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::poroelastic>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const PointSourceType &point_source,
    const PointPropertiesType &point_properties) {

  using PointAccelerationType =
      specfem::point::field<PointPropertiesType::dimension_tag,
                            PointPropertiesType::medium_tag, false, false, true,
                            false, PointPropertiesType::simd::using_simd>;

  PointAccelerationType result;

  result.acceleration(0) =
      point_source.stf(0) * point_source.lagrange_interpolant(0) *
      (1.0 - point_properties.phi() / point_properties.tortuosity());
  result.acceleration(1) =
      point_source.stf(1) * point_source.lagrange_interpolant(1) *
      (1.0 - point_properties.phi() / point_properties.tortuosity());
  result.acceleration(2) =
      point_source.stf(2) * point_source.lagrange_interpolant(2) *
      (1.0 - point_properties.rho_f() / point_properties.rho_bar());
  result.acceleration(3) =
      point_source.stf(3) * point_source.lagrange_interpolant(3) *
      (1.0 - point_properties.rho_f() / point_properties.rho_bar());

  return result;
}

} // namespace specfem::medium
