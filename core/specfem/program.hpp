#pragma once

#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks.hpp"
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

/**
 * @namespace specfem::program
 * @brief Core program execution and lifecycle management
 *
 * Provides unified execution interface for 2D and 3D simulations, runtime
 * context management (Kokkos/MPI initialization), and program abort utilities.
 */
namespace specfem::program {

/**
 * @brief Execute SPECFEM simulation with runtime dimension selection
 *
 * This function is the main entry point that the specfem executable calls. It
 * will call the appropriate internal program function based on the specified
 * dimension ("2d" or "3d"). And then run an entire simulation workflow
 * including mesh reading, source/receiver setup, assembly, time-stepping, and
 * output writing given the parameter configuration.
 *
 * @param dimension Dimension string ("2d" or "3d")
 * @param parameter_dict YAML parameter configuration
 * @param default_dict YAML default configuration
 * @return true if successful, false otherwise
 */
bool execute(const std::string &dimension, const YAML::Node &parameter_dict,
             const YAML::Node &default_dict);

} // namespace specfem::program

#include "program/abort.hpp"
#include "program/context.hpp"
