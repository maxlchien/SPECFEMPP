#pragma once

#include "impl/control_nodes.hpp"
#include "impl/points.hpp"
#include "impl/shape_functions.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <vector>

namespace specfem::assembly {
/**
 * @brief Information on an assembled mesh
 *
 */
template <>
struct mesh<specfem::dimension::type::dim3>
    : public specfem::assembly::mesh_impl::points<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::control_nodes<
          specfem::dimension::type::dim3>,
      public specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim3> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension
  int nspec;                          ///< Number of spectral
                                      ///< elements
  int ngnod;                          ///< Number of control
                                      ///< nodes

  specfem::mesh_entity::element<dimension_tag> element_grid; ///< Element number
                                                             ///< of GLL points

  mesh() = default;

  mesh(const specfem::mesh::parameters<dimension_tag> &parameters,
       const specfem::mesh::coordinates<dimension_tag> &coordinates,
       const specfem::mesh::mapping<dimension_tag> &mapping,
       const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
       const specfem::quadrature::quadratures &quadrature);
};

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
              specfem::dimension::type::dim3> &quadrature,
          ViewType &element_quadrature) {

  constexpr bool store_hprime_gll = ViewType::store_hprime_gll;

  constexpr bool store_weight_times_hprime_gll =
      ViewType::store_weight_times_hprime_gll;
  constexpr int NGLL = ViewType::ngll;

  static_assert(std::is_same_v<typename MemberType::execution_space,
                               Kokkos::DefaultExecutionSpace>,
                "Calling team must have a host execution space");

  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team, NGLL * NGLL * NGLL), [&](const int &xyz) {
        int ix, iz;
        sub2ind(xz, NGLL, iz, iy, ix);
        if constexpr (store_hprime_gll) {
          if constexpr (on_device) {
            element_quadrature.hprime_gll(iz, iy, ix) =
                quadrature.hprime(iz, iy, ix);
          } else {
            element_quadrature.hprime_gll(iz, iy, ix) =
                quadrature.h_hprime(iz, iy, ix);
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
template <typename MemberType, typename ViewType>
KOKKOS_FUNCTION void
load_on_device(const MemberType &team,
               const specfem::assembly::mesh_impl::quadrature<
                   specfem::dimension::type::dim2> &quadrature,
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
template <typename MemberType, typename ViewType>
void load_on_host(const MemberType &team,
                  const specfem::assembly::mesh_impl::quadrature<
                      specfem::dimension::type::dim2> &quadrature,
                  ViewType &element_quadrature) {
  impl_load<false>(team, quadrature, element_quadrature);
}

} // namespace specfem::assembly
