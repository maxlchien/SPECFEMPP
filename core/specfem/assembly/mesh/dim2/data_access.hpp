#pragma once

#include <Kokkos_Core.hpp>

namespace specfem::assembly {
/**
 * @defgroup QuadratureDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on host or device
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param mesh Mesh data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <bool on_device, typename MemberType, typename ViewType>
KOKKOS_INLINE_FUNCTION void
impl_load(const MemberType &team,
          const specfem::assembly::mesh_impl::quadrature<
              specfem::dimension::type::dim2> &quadrature,
          ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;

  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL), [&](const int &xz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, ix);
        if constexpr (store_hprime_gll) {
          if constexpr (on_device) {
            element_quadrature.hprime_gll(iz, ix) = quadrature.hprime(iz, ix);
          } else {
            element_quadrature.hprime_gll(iz, ix) = quadrature.h_hprime(iz, ix);
          }
        }
        if constexpr (store_weight_times_hprime_gll) {
          if constexpr (on_device) {
            element_quadrature.hprime_wgll(ix, iz) =
                quadrature.hprime(iz, ix) * quadrature.weights(iz);
          } else {
            element_quadrature.hprime_wgll(ix, iz) =
                quadrature.h_hprime(iz, ix) * quadrature.h_weights(iz);
          }
        }
      });
}

/**
 * @defgroup QuadratureDataAccess
 *
 */

/**
 * @brief Load quadrature data for a spectral element on the device
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param mesh Mesh data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType,
          typename std::enable_if_t<
              (specfem::data_access::is_element<ViewType>::value), int> = 0>
KOKKOS_FUNCTION void load_on_device(
    const MemberType &team,
    const specfem::assembly::mesh_impl::quadrature<ViewType::dimension_tag>
        &quadrature,
    ViewType &element_quadrature) {

  impl_load<true>(team, quadrature, element_quadrature);
}

/**
 * @brief Load quadrature data for a spectral element on the host
 *
 * @ingroup QuadratureDataAccess
 *
 * @tparam MemberType Member type. Needs to be a Kokkos::TeamPolicy member type
 * @tparam ViewType View type. Needs to be of @ref specfem::element::quadrature
 * @param team Team member
 * @param mesh Mesh data
 * @param element_quadrature Quadrature data for the element (output)
 */
template <typename MemberType, typename ViewType,
          typename std::enable_if_t<
              (specfem::data_access::is_element<ViewType>::value), int> = 0>
void load_on_host(
    const MemberType &team,
    const specfem::assembly::mesh_impl::quadrature<ViewType::dimension_tag>
        &quadrature,
    ViewType &element_quadrature) {
  impl_load<false>(team, quadrature, element_quadrature);
}
} // namespace specfem::assembly
