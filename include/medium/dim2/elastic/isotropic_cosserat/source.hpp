#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename PointSourcesType, typename PointPropertiesType>
KOKKOS_INLINE_FUNCTION auto impl_compute_source_contribution(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv_t>,
    const std::integral_constant<
        specfem::element::property_tag,
        specfem::element::property_tag::isotropic_cosserat>,
    const PointSourcesType &point_source,
    const PointPropertiesType &point_properties) {

  using PointAccelerationType =
      specfem::point::field<PointPropertiesType::dimension_tag,
                            PointPropertiesType::medium_tag, false, false, true,
                            false, PointPropertiesType::simd::using_simd>;

  PointAccelerationType result;

  result.acceleration(0) =
      point_source.stf(0) * point_source.lagrange_interpolant(0);
  result.acceleration(1) =
      point_source.stf(1) * point_source.lagrange_interpolant(1);
  result.acceleration(2) =
      point_source.stf(2) * point_source.lagrange_interpolant(2);

  return result;
}

} // namespace medium
} // namespace specfem
