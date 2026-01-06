#pragma once

/**
 * @brief Solver algorithms for time-domain wave propagation
 *
 * Contains base solver interfaces and implementations for explicit
 * time-stepping schemes used in spectral element wave simulations.
 */
namespace specfem::solver {}

#include "specfem/solver/solver.hpp"
#include "specfem/solver/time_marching.hpp"
#include "specfem/solver/time_marching.tpp"
