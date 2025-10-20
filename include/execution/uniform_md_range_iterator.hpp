#pragma once

#include "datatypes/impl/register_array.hpp"
#include "macros.hpp"
#include "policy.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace execution {

/**
 * @brief UniformMDRangeIterator class is used to iterate over all GLL points
 * within the given element.
 *
 * @tparam Rank The dimension of the iterator (2 or 3).
 * @tparam TeamMemberType The type of the Kokkos team member.
 *
 */
template <int Rank, typename TeamMemberType>
class UniformMDRangeIterator
    : public TeamThreadRangePolicy<TeamMemberType, int> {
private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  constexpr static auto rank = Rank; ///< Rank of the iterator (2 or 3)
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using policy_index_type =
      typename base_type::policy_index_type; /// Policy index type. Must be
                                             ///< convertible to integral type.
  using index_type = specfem::datatype::impl::RegisterArray<
      int, Kokkos::extents<std::size_t, rank>,
      Kokkos::layout_left>; ///< Underlying index type. This
                            ///< index will be passed to the
                            ///< closure when calling @ref
                            ///< kokkos::parallel_for with
                            ///< this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  template <int D = rank>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<D == 2, const index_type>::type
  operator()(const int &i) const {
    const int iz = i / n;
    const int ix = i % n;
    return index_type(iz, ix);
  }

  template <int D = rank>
  KOKKOS_INLINE_FUNCTION typename std::enable_if<D == 3, const index_type>::type
  operator()(const int &i) const {
    const int iz = i / (n * n);
    const int iy = (i / n) % n;
    const int ix = i % n;
    return index_type(iz, iy, ix);
  }

  /**
   * @brief Constructor for UniformMDRangeIterator.
   *
   * @param team The Kokkos team member type.
   * @param indices View of indices of elements within this chunk.
   * @param element_grid Element grid information containing n, ngllx, ngllz,
   * etc.
   */

  template <int D = rank, typename std::enable_if_t<D == 2, int> = 0>
  KOKKOS_INLINE_FUNCTION UniformMDRangeIterator(const TeamMemberType &team,
                                                int n)
      : n(n), base_type(team, n * n) {}

  template <int D = rank, typename std::enable_if_t<D == 3, int> = 0>
  KOKKOS_INLINE_FUNCTION UniformMDRangeIterator(const TeamMemberType &team,
                                                int n)
      : n(n), base_type(team, n * n * n) {}

private:
  int n; ///< Number of points in one direction within the iterator.
};

} // namespace execution
} // namespace specfem
