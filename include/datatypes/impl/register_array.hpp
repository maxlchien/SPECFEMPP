#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <mdspan/mdspan.hpp>
#include <sstream>
#include <type_traits>

namespace specfem {
namespace datatype {
namespace impl {

template <typename Extents> constexpr size_t compute_size() {

  size_t size = 1;
  for (size_t i = 0; i < Extents::rank(); ++i) {
    size *= Extents::static_extent(i);
  }
  return size;
}

template <typename Extents, typename... IndexType>
KOKKOS_INLINE_FUNCTION constexpr bool check_bounds(const IndexType &...i) {
  std::size_t index = 0;
  return ((i >= 0 && i < Extents::static_extent(index++)) && ...);
}

/**
 * @brief Class to represent a register array with a specified layout
 *
 * This class represents a register array with a specified layout. It is
 * templated on the type of the elements, the extents of the array, and the
 * layout.
 *
 * @tparam T Type of the elements in the array
 * @tparam Extents Extents of the array
 * @tparam Layout Layout of the array
 */
template <typename T, typename Extents, typename Layout> class RegisterArray {

private:
  constexpr static std::size_t rank = Extents::rank();
  constexpr static std::size_t size = impl::compute_size<Extents>();
  using mapping = typename Layout::template mapping<Extents>;

public:
  using value_type = T;

  KOKKOS_INLINE_FUNCTION
  RegisterArray(const value_type value) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = value;
    }
  }

  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == size, bool>::type = true>
  KOKKOS_INLINE_FUNCTION RegisterArray(const Args &...args)
      : m_value{ static_cast<value_type>(args)... } {}

  KOKKOS_INLINE_FUNCTION
  RegisterArray(const RegisterArray &other) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = other.m_value[i];
    }
  }

  KOKKOS_INLINE_FUNCTION
  RegisterArray() {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = 0.0;
    }
  }

  KOKKOS_INLINE_FUNCTION
  RegisterArray(const T *values) {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = values[i];
    }
  }

  template <typename... IndexType>
  KOKKOS_INLINE_FUNCTION constexpr value_type &
  operator()(const IndexType &...i) {
#ifndef NDEBUG
    // check if the indices are within bounds
    if (!check_bounds<Extents>(i...)) {
      // Abort the program with an error message
      Kokkos::abort("Index out of bounds");
    }
#endif
    return m_value[mapping()(i...)];
  }

  template <typename... IndexType>
  KOKKOS_INLINE_FUNCTION constexpr const value_type &
  operator()(const IndexType &...i) const {
#ifndef NDEBUG
    // check if the indices are within bounds
    if (!check_bounds<Extents>(i...)) {
      // Abort the program with an error message
      Kokkos::abort("Index out of bounds");
    }
#endif
    return m_value[mapping()(i...)];
  }

  KOKKOS_INLINE_FUNCTION
  T l2_norm() const {
    return l2_norm(std::integral_constant<bool, rank == 1>());
  }

  template <typename U = T,
            std::enable_if_t<std::is_integral<U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (m_value[i] != other.m_value[i]) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<std::is_floating_point<U>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::abs(m_value[i] - other.m_value[i]) >
          1e-6 * Kokkos::abs(m_value[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<std::is_integral<typename U::value_type>::value,
                             int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::Experimental::any_of(m_value[i] != other.m_value[i])) {
        return false;
      }
    }
    return true;
  }

  template <typename U = T,
            std::enable_if_t<
                std::is_floating_point<typename U::value_type>::value, int> = 0>
  KOKKOS_INLINE_FUNCTION bool operator==(const RegisterArray &other) const {
    for (std::size_t i = 0; i < size; ++i) {
      if (Kokkos::Experimental::any_of(
              Kokkos::abs(m_value[i] - other.m_value[i]) >
              1e-6 * Kokkos::abs(m_value[i]))) {
        return false;
      }
    }
    return true;
  }

  KOKKOS_INLINE_FUNCTION bool operator!=(const RegisterArray &other) const {
    return !(*this == other);
  }

private:
  T m_value[size]; ///< Data array

  KOKKOS_INLINE_FUNCTION
  T l2_norm(const std::true_type &) const {
    T norm = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
      norm += m_value[i] * m_value[i];
    }
    return Kokkos::sqrt(norm);
  }

  KOKKOS_INLINE_FUNCTION
  T l2_norm(const std::false_type &) const {
    static_assert(rank == 1, "l2_norm is only implemented for 1-D arrays");
    return 0.0;
  }
};

} // namespace impl
} // namespace datatype
} // namespace specfem
