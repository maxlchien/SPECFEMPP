#pragma once

#include "datatypes/impl/register_array.hpp"
#include "macros.hpp"
#include "policy.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace execution {

template <typename TeamMemberType, int... Ranks>
class TeamThreadMDRangeIterator;

template <typename TeamMemberType, int N1, int N2>
class TeamThreadMDRangeIterator<TeamMemberType, N1, N2>
    : public TeamThreadRangePolicy<TeamMemberType, int> {

private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  constexpr static auto rank = 2; ///< Rank of the iterator (2 or 3)
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using index_type = specfem::datatype::impl::RegisterArray<
      int, Kokkos::extents<std::size_t, rank>,
      Kokkos::layout_left>; ///< Underlying index type. This
                            ///< index will be passed to the
                            ///< closure when calling @ref
                            ///< kokkos::parallel_for with
                            ///< this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.

  KOKKOS_INLINE_FUNCTION const index_type operator()(const int &i) const {
   if constexpr (std::is_same_v<TeamMemberType::execution_space, Kokkos::DefaultHostExecutionSpace>) {
     const int i1 = i % N1;
     const int i2 = i / N1;
     return index_type(i1, i2);
   } else {
     const int i1 = i / N2;
     const int i2 = i % N2;
     return index_type(i1, i2);
   }
  }

  /**
   * @brief Constructor for TeamThreadMDRangeIterator.
   *
   * @param team The Kokkos team member type.
   * @param n Number of threads at each rank.
   */

  KOKKOS_INLINE_FUNCTION TeamThreadMDRangeIterator(const TeamMemberType &team)
      : base_type(team, N1 * N2) {}
};

template <typename TeamMemberType, int N1, int N2, int N3>
class TeamThreadMDRangeIterator<TeamMemberType, N1, N2, N3>
    : public TeamThreadRangePolicy<TeamMemberType, int> {

private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;

public:
  constexpr static auto rank = 3; ///< Rank of the iterator (2 or 3)
  using base_policy_type =
      typename base_type::base_policy_type; ///< Base policy type. Evaluates to
                                            ///< @c Kokkos::TeamThreadRange
  using index_type = specfem::datatype::impl::RegisterArray<
      int, Kokkos::extents<std::size_t, rank>,
      Kokkos::layout_left>; ///< Underlying index type. This
                            ///< index will be passed to the
                            ///< closure when calling @ref
                            ///< kokkos::parallel_for with
                            ///< this iterator.

  using execution_space =
      typename base_type::execution_space; ///< Execution space type.
  KOKKOS_INLINE_FUNCTION const index_type operator()(const int &i) const {
    const int i1 = i / (N2 * N3);
    const int i2 = (i / N3) % N2;
    const int i3 = i % N3;
    return index_type(i1, i2, i3);
  }

  KOKKOS_INLINE_FUNCTION TeamThreadMDRangeIterator(const TeamMemberType &team)
      : base_type(team, N1 * N2 * N3) {}

private:
  int n; ///< Number of points in one direction within the iterator.
};

} // namespace execution
} // namespace specfem
