#pragma once

#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Struct to store the assembled index for a quadrature point
 *
 * @tparam using_simd Flag to indicate if this is a simd index
 */
template <specfem::dimension::type DimensionTag> struct gll_index;

/**
 * @brief Struct to store the assembled index for a quadrature point
 *
 * This struct stores a 1D index that corresponds to a global numbering of the
 * quadrature point within the mesh.
 *
 */
template <>
struct gll_index<specfem::dimension::type::dim2>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::gll_index,
          specfem::dimension::type::dim2, false> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension tag
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  gll_index() = default;

  /**
   * @brief Constructor with values
   *
   * @param iglob Global index number of the quadrature point
   */
  KOKKOS_FUNCTION
  gll_index(const int &iz, const int &ix) : iz(iz), ix(ix) {}
  ///@}
};

template <>
struct gll_index<specfem::dimension::type::dim3>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::gll_index,
          specfem::dimension::type::dim3, false> {
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag
  int iz; ///< Index of the quadrature point in the z direction within
          ///< the spectral element
  int iy; ///< Index of the quadrature point in the y direction within
          ///< the spectral element
  int ix; ///< Index of the quadrature point in the x direction within
          ///< the spectral element

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  gll_index() = default;

  /**
   * @brief Constructor with values
   *
   * @param iglob Global index number of the quadrature point
   */
  KOKKOS_FUNCTION
  gll_index(const int &iz, const int &iy, const int &ix)
      : iz(iz), iy(iy), ix(ix) {}
  ///@}
};

} // namespace point
} // namespace specfem
