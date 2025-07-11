#ifndef _COMPUTE_PARTIAL_DERIVATIVES_HPP
#define _COMPUTE_PARTIAL_DERIVATIVES_HPP

#include "compute/compute_mesh.hpp"
#include "domain_view.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "macros.hpp"
#include "quadrature/interface.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {
/**
 * @brief Partial derivatives of the basis functions at every quadrature point
 *
 */
struct partial_derivatives : public specfem::container::Container<
                                 specfem::container::type::domain,
                                 specfem::data_class::type::partial_derivatives,
                                 specfem::dimension::type::dim2> {
  /**
   * @name Typedefs
   *
   */
  ///@{
  using base_type = specfem::container::Container<
      specfem::container::type::domain,
      specfem::data_class::type::partial_derivatives,
      specfem::dimension::type::dim2>; ///< Base type of the point partial
                                       ///< derivatives
  using view_type = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  ///@}

  int nspec; ///< Number of spectral elements
  int ngllz; ///< Number of quadrature points in z direction
  int ngllx; ///< Number of quadrature points in x direction

  view_type xix;                    ///< @xix
  view_type::HostMirror h_xix;      ///< Host mirror of @xix
  view_type xiz;                    ///< @xiz
  view_type::HostMirror h_xiz;      ///< Host mirror of @xiz
  view_type gammax;                 ///< @gammax
  view_type::HostMirror h_gammax;   ///< Host mirror of @gammax
  view_type gammaz;                 ///< @gammaz
  view_type::HostMirror h_gammaz;   ///< Host mirror of @gammaz
  view_type jacobian;               ///< Jacobian
  view_type::HostMirror h_jacobian; ///< Host mirror of Jacobian

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  partial_derivatives() = default;

  partial_derivatives(const int nspec, const int ngllz, const int ngllx);
  /**
   * @brief Construct a new partial derivatives object from mesh information
   *
   * @param mesh Mesh information
   */
  partial_derivatives(const specfem::compute::mesh &mesh);
  ///@}

  void sync_views();

  /**
   * @brief Check if the Jacobian is a small value
   *
   * @return std::tuple<bool, Kokkos::View> Tuple containing a boolean
   * indicating whether a small Jacobian was found and a view containing the
   * indices of the spectral elements with small Jacobian
   */
  std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
  check_small_jacobian() const;
};

/**
 * @defgroup ComputePartialDerivativesDataAccess
 *
 * @brief Functions to load and store partial derivatives at a given quadrature
 * point
 *
 */

template <bool on_device, typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::simd_index<PointPartialDerivativesType::dimension_tag>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  using simd = typename PointPartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  if constexpr (on_device) {
    Kokkos::Experimental::where(mask, partial_derivatives.xix)
        .copy_from(&derivatives.xix[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammax)
        .copy_from(&derivatives.gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.xiz)
        .copy_from(&derivatives.xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
        .copy_from(&derivatives.gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
          .copy_from(&derivatives.jacobian[_index], tag_type());
    }
  } else {
    Kokkos::Experimental::where(mask, partial_derivatives.xix)
        .copy_from(&derivatives.h_xix[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammax)
        .copy_from(&derivatives.h_gammax[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.xiz)
        .copy_from(&derivatives.h_xiz[_index], tag_type());
    Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
        .copy_from(&derivatives.h_gammaz[_index], tag_type());
    if constexpr (StoreJacobian) {
      Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
          .copy_from(&derivatives.h_jacobian[_index], tag_type());
    }
  }
}

template <bool on_device, typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::simd::using_simd, int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void impl_load(
    const specfem::point::index<PointPartialDerivativesType::dimension_tag>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  if constexpr (on_device) {
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xix = derivatives.xix.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammax = derivatives.gammax.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        xiz = derivatives.xiz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        gammaz = derivatives.gammaz.get_base_view();
    Kokkos::View<const type_real *, Kokkos::DefaultExecutionSpace::memory_space,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess> >
        jacobian = derivatives.jacobian.get_base_view();

    partial_derivatives.xix = xix(_index);
    partial_derivatives.gammax = gammax(_index);
    partial_derivatives.xiz = xiz(_index);
    partial_derivatives.gammaz = gammaz(_index);
    if constexpr (StoreJacobian) {
      partial_derivatives.jacobian = jacobian(_index);
    }
  } else {
    partial_derivatives.xix = derivatives.h_xix[_index];
    partial_derivatives.gammax = derivatives.h_gammax[_index];
    partial_derivatives.xiz = derivatives.h_xiz[_index];
    partial_derivatives.gammaz = derivatives.h_gammaz[_index];
    if constexpr (StoreJacobian) {
      partial_derivatives.jacobian = derivatives.h_jacobian[_index];
    }
  }
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              PointPartialDerivativesType::simd::using_simd, int> = 0>
inline void impl_store_on_host(
    const specfem::point::simd_index<PointPartialDerivativesType::dimension_tag>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int nspec = derivatives.nspec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  using simd = typename PointPartialDerivativesType::simd;
  using mask_type = typename simd::mask_type;
  using tag_type = typename simd::tag_type;

  mask_type mask([&](std::size_t lane) { return index.mask(lane); });

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  Kokkos::Experimental::where(mask, partial_derivatives.xix)
      .copy_to(&derivatives.h_xix[_index], tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammax)
      .copy_to(&derivatives.h_gammax[_index], tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.xiz)
      .copy_to(&derivatives.h_xiz[_index], tag_type());
  Kokkos::Experimental::where(mask, partial_derivatives.gammaz)
      .copy_to(&derivatives.h_gammaz[_index], tag_type());
  if constexpr (StoreJacobian) {
    Kokkos::Experimental::where(mask, partial_derivatives.jacobian)
        .copy_to(&derivatives.h_jacobian[_index], tag_type());
  }
}

template <typename PointPartialDerivativesType,
          typename std::enable_if_t<
              !PointPartialDerivativesType::simd::using_simd, int> = 0>
inline void impl_store_on_host(
    const specfem::point::index<PointPartialDerivativesType::dimension_tag>
        &index,
    const specfem::compute::partial_derivatives &derivatives,
    const PointPartialDerivativesType &partial_derivatives) {

  const int ispec = index.ispec;
  const int iz = index.iz;
  const int ix = index.ix;

  constexpr static bool StoreJacobian =
      PointPartialDerivativesType::store_jacobian;

  const auto &mapping = derivatives.xix.get_mapping();
  const std::size_t _index = mapping(ispec, iz, ix);

  derivatives.h_xix[_index] = partial_derivatives.xix;
  derivatives.h_gammax[_index] = partial_derivatives.gammax;
  derivatives.h_xiz[_index] = partial_derivatives.xiz;
  derivatives.h_gammaz[_index] = partial_derivatives.gammaz;
  if constexpr (StoreJacobian) {
    derivatives.h_jacobian[_index] = partial_derivatives.jacobian;
  }
}

/**
 * @brief Load the partial derivatives at a given quadrature point on the device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
KOKKOS_FORCEINLINE_FUNCTION void
load_on_device(const IndexType &index,
               const specfem::compute::partial_derivatives &derivatives,
               PointPartialDerivativesType &partial_derivatives) {
  impl_load<true>(index, derivatives, partial_derivatives);
}

/**
 * @brief Store the partial derivatives at a given quadrature point on the
 * device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
inline void
load_on_host(const IndexType &index,
             const specfem::compute::partial_derivatives &derivatives,
             PointPartialDerivativesType &partial_derivatives) {
  impl_load<false>(index, derivatives, partial_derivatives);
}

/**
 * @brief Store the partial derivatives at a given quadrature point on the
 * device
 *
 * @ingroup ComputePartialDerivativesDataAccess
 *
 * @tparam PointPartialDerivativesType Point partial derivatives type. Needs to
 * be of @ref specfem::point::partial_derivatives
 * @tparam IndexType Index type. Needs to be of @ref specfem::point::index or
 * @ref specfem::point::simd_index
 * @param index Index of the quadrature point
 * @param derivatives Partial derivatives container
 * @param partial_derivatives Partial derivatives at the given quadrature point
 */
template <
    typename PointPartialDerivativesType, typename IndexType,
    typename std::enable_if_t<IndexType::using_simd ==
                                  PointPartialDerivativesType::simd::using_simd,
                              int> = 0>
inline void
store_on_host(const IndexType &index,
              const specfem::compute::partial_derivatives &derivatives,
              const PointPartialDerivativesType &partial_derivatives) {
  impl_store_on_host(index, derivatives, partial_derivatives);
}
} // namespace compute
} // namespace specfem

#endif
