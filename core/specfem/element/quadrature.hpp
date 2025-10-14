#pragma once

#include "datatypes/element_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace element {
/**
 * @brief Struct used to store quadrature values within an element.
 *
 * Currently we store the derivatives of the Lagrange polynomials since these
 * are the variables required when computing gradients and divergences in
 * compute_forces.
 *
 * @tparam NGLL Number of Gauss-Lobatto-Legendre points
 * @tparam DimensionTag Dimension of the element
 * @tparam MemorySpace Memory space for the views
 * @tparam MemoryTraits Memory traits for the views
 */
template <int NGLL, specfem::dimension::type DimensionTag, typename MemorySpace,
          typename MemoryTraits>
struct quadrature : public specfem::data_access::Accessor<
                        specfem::data_access::AccessorType::element,
                        specfem::data_access::DataClassType::quadrature,
                        DimensionTag, false> {
  /**
   * @name Typedefs
   *
   */
  ///@{

  /**
   * @brief Underlying view type used to store quadrature values.
   *
   */
  using ViewType =
      specfem::datatype::ScalarElementViewType<type_real,
                                               specfem::dimension::type::dim2,
                                               NGLL, MemorySpace, MemoryTraits>;
  ViewType hprime_gll; ///< Derivatives of lagrange polynomials \f$l'\f$ at GLL
                       ///< points.
  constexpr static auto dimension_tag =
      DimensionTag;                 ///< Dimension tag (dim2 or dim3)
  constexpr static int ngll = NGLL; ///< Number of GLL points
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION quadrature() = default;

  /**
   * @brief Constructor that initializes the quadrature with a view.
   *
   * @param view View to initialize the quadrature with.
   */
  KOKKOS_FUNCTION quadrature(const ViewType &view) : hprime_gll(view) {}

  /**
   * @brief Constructor that initializes the quadrature within Scratch Memory.
   *
   * @tparam MemberType Type of the Kokkos team member.
   * @param team Kokkos team member.
   */
  template <typename MemberType>
  KOKKOS_FUNCTION quadrature(const MemberType &team)
      : hprime_gll(team.team_scratch(0)) {
    static_assert(
        Kokkos::SpaceAccessibility<typename MemberType::execution_space,
                                   MemorySpace>::accessible,
        "MemorySpace is not accessible from the execution space");
  }
  ///@}
  /**
   * @brief Get the amount memory in bytes required for shared memory
   *
   * @return int Amount of shared memory in bytes
   */
  constexpr static int shmem_size() { return ViewType::shmem_size(); }
};

} // namespace element
} // namespace specfem
