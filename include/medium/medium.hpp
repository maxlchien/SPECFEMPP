#pragma once

/**
 * @brief Physics computations for seismic wave propagation media.
 *
 * Provides computational functions for acoustic, elastic, poroelastic,
 * electromagnetic, and cosserat media in 2D/3D with isotropic/anisotropic
 * properties. Uses template dispatch for compile-time medium selection.
 *
 * **Core functions:**
 * - `compute_stress()`: Stress tensor from field derivatives
 * - `compute_wavefield()`: Wavefield from intrinsic fields
 * - `compute_source_contribution()`: Source terms
 * - `compute_frechet_derivatives()`: Sensitivity kernels
 * - `material<>`, `properties_container<>`: Material property management
 */
namespace specfem::medium {}

#include "compute_frechet_derivatives.hpp"
#include "compute_stress.hpp"
#include "compute_wavefield.hpp"
