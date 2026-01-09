#pragma once
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::compute_source_array_impl {

/**
 * @brief Compute source array for a 2D vector source using Lagrange
 * interpolation.
 *
 * Evaluates Lagrange basis functions at the source location and
 * distributes force vector components to GLL quadrature points:
 * @f$ S_{i,jz,jx} = L_{jx}(\xi_s) L_{jz}(\gamma_s) f_i @f$
 *
 * @param source Vector source containing force components and local coordinates
 * @param source_array Output array of shape (ncomponents, ngllz, ngllx)
 *
 * @note source_array values are non-zero only at the source's element location.
 * The magnitude at each GLL point equals the force vector component scaled by
 * the Lagrange interpolant product.
 */
void from_vector(
    const specfem::sources::vector_source<specfem::dimension::type::dim2>
        &source,
    Kokkos::View<type_real ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        source_array);

} // namespace specfem::assembly::compute_source_array_impl
