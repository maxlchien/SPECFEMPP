#pragma once

#include "enumerations/medium.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION
    specfem::point::stress<specfem::dimension::type::dim2,
                           specfem::element::medium_tag::elastic_psv, UseSIMD>
    impl_compute_stress(
        const specfem::point::properties<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv,
            specfem::element::property_tag::isotropic, UseSIMD> &properties,
        const specfem::point::field_derivatives<
            specfem::dimension::type::dim2,
            specfem::element::medium_tag::elastic_psv, UseSIMD>
            &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_zz, sigma_xz;

  // P_SV case
  // sigma_xx
  sigma_xx =
      properties.lambdaplus2mu() * du(0, 0) + properties.lambda() * du(1, 1);

  // sigma_zz
  sigma_zz =
      properties.lambdaplus2mu() * du(1, 1) + properties.lambda() * du(0, 0);

  // sigma_xz
  sigma_xz = properties.mu() * (du(0, 1) + du(1, 0));

  specfem::datatype::VectorPointViewType<type_real, 2, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;
  T(1, 0) = sigma_xz;
  T(1, 1) = sigma_zz;

  return { T };
}

template <bool UseSIMD>
KOKKOS_INLINE_FUNCTION specfem::point::stress<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sh,
    UseSIMD>
impl_compute_stress(
    const specfem::point::properties<specfem::dimension::type::dim2,
                                     specfem::element::medium_tag::elastic_sh,
                                     specfem::element::property_tag::isotropic,
                                     UseSIMD> &properties,
    const specfem::point::field_derivatives<
        specfem::dimension::type::dim2,
        specfem::element::medium_tag::elastic_sh, UseSIMD> &field_derivatives) {

  using datatype =
      typename specfem::datatype::simd<type_real, UseSIMD>::datatype;
  const auto &du = field_derivatives.du;

  datatype sigma_xx, sigma_xz;

  // SH-case
  // sigma_xx
  sigma_xx = properties.mu() * du(0, 0); // would be sigma_xy in
                                         // CPU-version

  // sigma_xz
  sigma_xz = properties.mu() * du(0, 1); // sigma_zy

  specfem::datatype::VectorPointViewType<type_real, 1, 2, UseSIMD> T;

  T(0, 0) = sigma_xx;
  T(0, 1) = sigma_xz;

  return { T };
}

} // namespace medium
} // namespace specfem
