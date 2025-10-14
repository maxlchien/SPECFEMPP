#pragma once

#include "macros.hpp"
#include "policy.hpp"
#include "specfem/point.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace execution {

/**
 * @brief ElementIterator class is used to iterate over all GLL points
 * within the given element.
 *
 * @tparam DimensionTag The dimension tag (e.g., dim2, dim3).
 * @tparam TeamMemberType The type of the Kokkos team member.
 *
 */
template <specfem::dimension::type DimensionTag, typename TeamMemberType>
class ElementIterator : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;
  constexpr static auto dimension_tag = DimensionTag;

public:
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using policy_index_type =
      typename base_type::policy_index_type; /// Policy index type. Must be
                                             ///< convertible to integral type.
  using index_type =
      specfem::point::gll_index<dimension_tag>; ///< Underlying index type. This
                                                ///< index will be passed to the
                                                ///< closure when calling @ref
                                                ///< kokkos::parallel_for with
                                                ///< this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  template <specfem::dimension::type D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<D == specfem::dimension::type::dim2,
                              const index_type>::type
      operator()(const int &i) const {
    const int iz = i / ngll;
    const int ix = i % ngll;
    return index_type(iz, ix);
  }

  template <specfem::dimension::type D = dimension_tag>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<D == specfem::dimension::type::dim3,
                              const index_type>::type
      operator()(const int &i) const {
    const int iz = i / (ngll * ngll);
    const int iy = (i / ngll) % ngll;
    const int ix = i % ngll;
    return index_type(iz, iy, ix);
  }

  /**
   * @brief Constructor for ElementIterator.
   *
   * @param team The Kokkos team member type.
   * @param indices View of indices of elements within this chunk.
   * @param element_grid Element grid information containing ngll, ngllx, ngllz,
   * etc.
   */

  template <
      specfem::dimension::type D = dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim2, int> = 0>
  KOKKOS_INLINE_FUNCTION ElementIterator(const TeamMemberType &team, int ngll)
      : ngll(ngll), base_type(team, ngll * ngll) {}

  template <
      specfem::dimension::type D = dimension_tag,
      typename std::enable_if_t<D == specfem::dimension::type::dim3, int> = 0>
  KOKKOS_INLINE_FUNCTION ElementIterator(const TeamMemberType &team, int ngll)
      : ngll(ngll), base_type(team, ngll * ngll * ngll) {}

private:
  int ngll; ///< Number of GLL points in one direction within the element
};

} // namespace execution
} // namespace specfem
