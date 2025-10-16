#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::assembly {

template <
    typename PointIndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<PointIndexType>::value &&
            PointIndexType::dimension_tag == specfem::dimension::type::dim3 &&
            !PointIndexType::using_simd && !PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const PointIndexType &index,
                                                const ContainerType &container,
                                                PointType &point) {
  static_assert(
      specfem::data_access::CheckCompatibility<PointIndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types");

  constexpr static bool load_jacobian = PointType::store_jacobian;

  point.xix = container.xix(index.ispec, index.iz, index.iy, index.ix);
  point.xiy = container.xiy(index.ispec, index.iz, index.iy, index.ix);
  point.xiz = container.xiz(index.ispec, index.iz, index.iy, index.ix);
  point.etax = container.etax(index.ispec, index.iz, index.iy, index.ix);
  point.etay = container.etay(index.ispec, index.iz, index.iy, index.ix);
  point.etaz = container.etaz(index.ispec, index.iz, index.iy, index.ix);
  point.gammax = container.gammax(index.ispec, index.iz, index.iy, index.ix);
  point.gammay = container.gammay(index.ispec, index.iz, index.iy, index.ix);
  point.gammaz = container.gammaz(index.ispec, index.iz, index.iy, index.ix);
  if constexpr (load_jacobian) {
    point.jacobian =
        container.jacobian(index.ispec, index.iz, index.iy, index.ix);
  }
}

// SIMD version
template <
    typename PointIndexType, typename ContainerType, typename PointType,
    typename std::enable_if_t<
        specfem::data_access::is_index_type<PointIndexType>::value &&
            PointIndexType::dimension_tag == specfem::dimension::type::dim3 &&
            PointIndexType::using_simd && PointType::simd::using_simd &&
            specfem::data_access::is_jacobian_matrix<ContainerType>::value,
        int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void load_on_device(const PointIndexType &index,
                                                const ContainerType &container,
                                                PointType &point) {
  static_assert(
      specfem::data_access::CheckCompatibility<PointIndexType, ContainerType,
                                               PointType>::value,
      "Incompatible types");

  constexpr static bool load_jacobian = PointType::store_jacobian;

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int iy = index.iy;
  const int ix = index.ix;

  using simd = typename PointType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  // Get data pointers from Kokkos::View
  const type_real *xix_ptr = container.xix.data();
  const type_real *xiy_ptr = container.xiy.data();
  const type_real *xiz_ptr = container.xiz.data();
  const type_real *etax_ptr = container.etax.data();
  const type_real *etay_ptr = container.etay.data();
  const type_real *etaz_ptr = container.etaz.data();
  const type_real *gammax_ptr = container.gammax.data();
  const type_real *gammay_ptr = container.gammay.data();
  const type_real *gammaz_ptr = container.gammaz.data();

  // Calculate linear index for LayoutLeft (column-major) 4D array
  // For LayoutLeft: index = ix + ngllx * (iy + nglly * (iz + ngllz * ispec))
  const int ngllx = container.xix.extent(3);
  const int nglly = container.xix.extent(2);
  const int ngllz = container.xix.extent(1);
  const std::size_t _index = ix + ngllx * (iy + nglly * (iz + ngllz * ispec));

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  Kokkos::Experimental::where(mask, point.xix)
      .copy_from(&xix_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.xiy)
      .copy_from(&xiy_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.xiz)
      .copy_from(&xiz_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.etax)
      .copy_from(&etax_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.etay)
      .copy_from(&etay_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.etaz)
      .copy_from(&etaz_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.gammax)
      .copy_from(&gammax_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.gammay)
      .copy_from(&gammay_ptr[_index], tag_type());
  Kokkos::Experimental::where(mask, point.gammaz)
      .copy_from(&gammaz_ptr[_index], tag_type());
  if constexpr (load_jacobian) {
    const type_real *jacobian_ptr = container.jacobian.data();
    Kokkos::Experimental::where(mask, point.jacobian)
        .copy_from(&jacobian_ptr[_index], tag_type());
  }
}

} // namespace specfem::assembly
