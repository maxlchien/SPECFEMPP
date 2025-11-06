#pragma once
/**
 * @file execute.hpp
 * @brief Unified SPECFEM++ execution interface
 *
 * This header defines the unified execution function for SPECFEM++ simulations,
 * supporting both 2D and 3D simulations through dimension-templated execution.
 *
 * The execute function initializes and runs the simulation based on the
 * provided parameters and periodic tasks.
 */
#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks.hpp"
#include <chrono>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

// Legacy execute function (currently used by existing code)
void execute(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi);

// Future templated execute functions (to be implemented)
// These will replace the legacy function once dimension-specific logic is
// refactored

template <specfem::dimension::type DIM>
void execute_templated(
    const YAML::Node &parameter_dict, const YAML::Node &default_dict,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> > tasks,
    specfem::MPI::MPI *mpi);

// Explicit declarations for template specializations
extern template void execute_templated<specfem::dimension::type::dim2>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >,
    specfem::MPI::MPI *);

extern template void execute_templated<specfem::dimension::type::dim3>(
    const YAML::Node &, const YAML::Node &,
    std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >,
    specfem::MPI::MPI *);
