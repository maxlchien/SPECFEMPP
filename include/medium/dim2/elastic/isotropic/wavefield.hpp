#pragma once

#include "algorithms/dot.hpp"
#include "algorithms/gradient.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_stress.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <typename ChunkIndexType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_psv>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  using FieldDerivativesType = specfem::point::field_derivatives<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      false>;

  using PointPropertyType = specfem::point::properties<
      specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_psv,
      specfem::element::property_tag::isotropic, false>;

  const auto &properties = assembly.properties;

  const auto &active_field = [&]() {
    if (wavefield_component == specfem::wavefield::type::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::type::velocity) {
      return field.velocity;
    } else if (wavefield_component == specfem::wavefield::type::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::type::pressure) {
      return field.displacement;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  if (wavefield_component == specfem::wavefield::type::pressure) {

    specfem::algorithms::gradient(
        chunk_index, assembly.partial_derivatives, quadrature.hprime_gll,
        active_field,
        [&](const typename ChunkIndexType::iterator_type::index_type
                &iterator_index,
            const FieldDerivativesType::value_type &du) {
          const auto index = iterator_index.get_index();
          const int ielement = iterator_index.get_policy_index();
          PointPropertyType point_property;

          specfem::compute::load_on_device(index, properties, point_property);

          // // P_SV case
          // // sigma_xx
          // const auto sigma_xx = point_property.lambdaplus2mu * du(0, 0) +
          //                       point_property.lambda * du(1, 1);

          // // sigma_zz
          // const auto sigma_zz = point_property.lambdaplus2mu * du(1, 1) +
          //                       point_property.lambda * du(0, 0);

          // // sigma_yy
          // const auto sigma_yy =
          //     point_property.lambda * (du(0, 0) + du(1, 1));

          // wavefield(iterator_index.ielement, index.iz, index.ix, 0) =
          //     -1.0 * (sigma_xx + sigma_zz + sigma_yy) / 3.0;
          wavefield(ielement, index.iz, index.ix, 0) =
              -1.0 *
              ((point_property.lambda() + (2.0 / 3.0) * point_property.mu()) *
               (du(0, 0) + du(1, 1)));
        });

    return;
  }

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        wavefield(ielement, index.iz, index.ix, 0) =
            active_field(ielement, index.iz, index.ix, 0);
        wavefield(ielement, index.iz, index.ix, 1) =
            active_field(ielement, index.iz, index.ix, 1);
      });

  return;
}

template <typename ChunkIndexType, typename ChunkFieldType,
          typename QuadratureType, typename WavefieldViewType>
KOKKOS_FUNCTION void impl_compute_wavefield(
    const std::integral_constant<specfem::dimension::type,
                                 specfem::dimension::type::dim2>,
    const std::integral_constant<specfem::element::medium_tag,
                                 specfem::element::medium_tag::elastic_sh>,
    const std::integral_constant<specfem::element::property_tag,
                                 specfem::element::property_tag::isotropic>,
    const ChunkIndexType &chunk_index,
    const specfem::compute::assembly &assembly,
    const QuadratureType &quadrature, const ChunkFieldType &field,
    const specfem::wavefield::type wavefield_component,
    WavefieldViewType wavefield) {

  if (wavefield_component == specfem::wavefield::type::pressure) {
    Kokkos::abort("pressure not supported for SH");

    return;
  }

  const auto &active_field = [&]() {
    if (wavefield_component == specfem::wavefield::type::displacement) {
      return field.displacement;
    } else if (wavefield_component == specfem::wavefield::type::velocity) {
      return field.velocity;
    } else if (wavefield_component == specfem::wavefield::type::acceleration) {
      return field.acceleration;
    } else if (wavefield_component == specfem::wavefield::type::rotation) {
      return field.displacement;
    } else {
      Kokkos::abort("component not supported");
    }
  }();

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename ChunkIndexType::iterator_type::index_type
              &iterator_index) {
        const auto index = iterator_index.get_index();
        const int ielement = iterator_index.get_policy_index();
        wavefield(ielement, index.iz, index.ix, 0) =
            active_field(ielement, index.iz, index.ix, 0);
      });

  return;
}

} // namespace medium
} // namespace specfem
