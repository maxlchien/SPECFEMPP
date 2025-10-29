#pragma once

#include "datatypes/impl/register_array.hpp"
#include "macros.hpp"
#include "policy.hpp"
#include "void_iterator.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem {
namespace execution {

namespace impl {
/**
 * @brief Base template for N-dimensional range iterator implementation details
 * @tparam Rank The dimensionality of the iterator
 */
template <int Rank> struct TeamThreadMDRangeIteratorN {};

/**
 * @brief Specialization for 2D range iterator
 */
template <> struct TeamThreadMDRangeIteratorN<2> {
  int n1_; ///< Size of first dimension
  int n2_; ///< Size of second dimension
  TeamThreadMDRangeIteratorN(int n1, int n2) : n1_(n1), n2_(n2) {}
};

/**
 * @brief Specialization for 3D range iterator
 */
template <> struct TeamThreadMDRangeIteratorN<3> {
  int n1_; ///< Size of first dimension
  int n2_; ///< Size of second dimension
  int n3_; ///< Size of third dimension
  TeamThreadMDRangeIteratorN(int n1, int n2, int n3)
      : n1_(n1), n2_(n2), n3_(n3) {}
};
} // namespace impl

/**
 * @brief Multi-dimensional range iterator for team-based parallel execution
 *
 * This class provides a way to iterate over multi-dimensional ranges in
 * parallel using Kokkos team-based execution. It supports both 2D and 3D
 * iterations and handles different execution spaces (host and device)
 * appropriately.
 *
 * @tparam TeamMemberType The Kokkos team member type
 * @tparam Rank The dimensionality of the iterator (2 or 3)
 */
template <typename TeamMemberType, int Rank>
class TeamThreadMDRangeIterator
    : public TeamThreadRangePolicy<TeamMemberType, int>,
      private impl::TeamThreadMDRangeIteratorN<Rank> {

private:
  using base_type = TeamThreadRangePolicy<TeamMemberType, int>;
  using n_type = impl::TeamThreadMDRangeIteratorN<Rank>;

public:
  constexpr static auto rank = Rank; ///< Rank of the iterator (2 or 3)
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

  /**
   * @brief Convert linear index to 2D / 3D coordinates
   *
   * @param i Linear index
   * @return index_type 2D / 3D coordinates
   */
  KOKKOS_INLINE_FUNCTION const index_type operator()(const int &i) const {
    if constexpr (Rank == 2) {
      const int &n1 = n_type::n1_;
      const int &n2 = n_type::n2_;
      if constexpr (std::is_same_v<typename TeamMemberType::execution_space,
                                   Kokkos::DefaultHostExecutionSpace>) {
        const int i1 = i % n1;
        const int i2 = i / n1;
        return index_type(i1, i2);
      } else {
        const int i1 = i / n2;
        const int i2 = i % n2;
        return index_type(i1, i2);
      }
    } else if constexpr (Rank == 3) {
      const int &n1 = n_type::n1_;
      const int &n2 = n_type::n2_;
      const int &n3 = n_type::n3_;
      if constexpr (std::is_same_v<typename TeamMemberType::execution_space,
                                   Kokkos::DefaultHostExecutionSpace>) {
        const int i1 = i % n1;
        const int i2 = (i / n1) % n2;
        const int i3 = i / (n1 * n2);
        return index_type(i1, i2, i3);
      } else {
        const int i1 = i / (n2 * n3);
        const int i2 = (i / n3) % n2;
        const int i3 = i % n3;
        return index_type(i1, i2, i3);
      }
    }
  }

  /**
   * @brief Constructor for 2D TeamThreadMDRangeIterator
   *
   * @param team The Kokkos team member
   * @param n1 Size of first dimension
   * @param n2 Size of second dimension
   */
  template <int R = Rank, typename std::enable_if_t<R == 2, int> = 0>
  KOKKOS_INLINE_FUNCTION TeamThreadMDRangeIterator(const TeamMemberType &team,
                                                   int n1, int n2)
      : base_type(team, n1 * n2), n_type{ n1, n2 } {}

  /**
   * @brief Constructor for 3D TeamThreadMDRangeIterator
   *
   * @param team The Kokkos team member
   * @param n1 Size of first dimension
   * @param n2 Size of second dimension
   * @param n3 Size of third dimension
   */
  template <int R = Rank, typename std::enable_if_t<R == 3, int> = 0>
  KOKKOS_INLINE_FUNCTION TeamThreadMDRangeIterator(const TeamMemberType &team,
                                                   int n1, int n2, int n3)
      : base_type(team, n1 * n2 * n3), n_type{ n1, n2, n3 } {}
};

// User-defined deduction guide
template <typename TeamMemberType>
TeamThreadMDRangeIterator(TeamMemberType, int, int)
    -> TeamThreadMDRangeIterator<TeamMemberType, 2>;

template <typename TeamMemberType>
TeamThreadMDRangeIterator(TeamMemberType, int, int, int)
    -> TeamThreadMDRangeIterator<TeamMemberType, 3>;

} // namespace execution
} // namespace specfem
