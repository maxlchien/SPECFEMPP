#pragma once

/**
 * @brief Data access utilities used within SPECFEM++
 *
 * This module provides containers, accessors, and compatibilty checkers that
 * help in managing different data access patterns we encounter within spectral
 * element simulations.
 */
namespace specfem::data_access {}

#include "data_access/accessor.hpp"
#include "data_access/check_compatibility.hpp"
#include "data_access/container.hpp"
#include "data_access/data_class.hpp"
