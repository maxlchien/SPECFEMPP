#pragma once

/**
 * @brief Chunk-based data accessors for edge operations in spectral elements
 *
 * Provides vectorized access to field data and interface coupling terms
 * for chunks of element edges in the spectral element method.
 *
 * Key components:
 * - Field accessors (displacement, velocity, acceleration)
 * - Nonconforming interface coupling data
 * - Index mapping for chunk operations
 */
namespace specfem::chunk_edge {}

#include "chunk_edge/acceleration.hpp"
#include "chunk_edge/displacement.hpp"
#include "chunk_edge/index.hpp"
#include "chunk_edge/nonconforming_interface.hpp"
#include "chunk_edge/velocity.hpp"
