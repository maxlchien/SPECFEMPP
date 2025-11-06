#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/execution.hpp"
#include "periodic_tasks/periodic_task.hpp"
#include "specfem_mpi/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem {

/**
 * @brief Unified SPECFEM++ context class for managing initialization,
 * execution, and finalization
 *
 * This class provides a clean singleton interface for managing Kokkos and MPI
 * initialization, dimension-templated execution, and proper resource cleanup.
 * It replaces the global variable approach with a more maintainable RAII-based
 * design.
 */
class Context {
public:
  /**
   * @brief Get the singleton instance
   */
  static Context &instance();

  /**
   * @brief Initialize Kokkos and MPI
   *
   * @param argc Command line argument count
   * @param argv Command line arguments
   * @return true if initialization successful
   */
  bool initialize(int argc, char *argv[]);

  /**
   * @brief Initialize from Python with argument list
   *
   * @param py_argv Python list of command line arguments
   * @return true if initialization successful
   */
  bool initialize_from_python(const std::vector<std::string> &py_argv);

  /**
   * @brief Finalize and cleanup all resources
   *
   * @return true if finalization successful
   */
  bool finalize();

  /**
   * @brief Execute simulation with dimension template
   *
   * @tparam DimensionTag Dimension type (dim2 or dim3)
   * @param parameter_dict YAML parameter configuration
   * @param default_dict YAML default configuration
   * @param tasks Vector of periodic tasks
   * @return true if execution successful
   */
  template <specfem::dimension::type DimensionTag>
  bool
  execute(const YAML::Node &parameter_dict, const YAML::Node &default_dict,
          std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
              &tasks);

  /**
   * @brief Execute simulation with dimension specified as string
   *
   * @param dimension Dimension string ("2d" or "3d")
   * @param parameter_dict YAML parameter configuration
   * @param default_dict YAML default configuration
   * @param tasks Vector of periodic tasks
   * @return true if execution successful
   */
  bool execute_with_dimension(
      const std::string &dimension, const YAML::Node &parameter_dict,
      const YAML::Node &default_dict,
      std::vector<std::shared_ptr<specfem::periodic_tasks::periodic_task> >
          &tasks);

  /**
   * @brief Get MPI instance (for backward compatibility)
   *
   * @return Pointer to MPI instance or nullptr if not initialized
   */
  specfem::MPI::MPI *get_mpi() const;

  /**
   * @brief Check if core is initialized
   *
   * @return true if initialized
   */
  bool is_initialized() const;

  /**
   * @brief Check if Kokkos is initialized
   *
   * @return true if Kokkos is initialized
   */
  bool is_kokkos_initialized() const;

  /**
   * @brief Destructor - ensures proper cleanup
   */
  ~Context();

private:
  /**
   * @brief Private constructor for singleton
   */
  Context();

  /**
   * @brief Delete copy constructor and assignment operator
   */
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  /**
   * @brief Internal helper to convert string arguments to argc/argv
   */
  void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                       char **&argv);
  void cleanup_argc_argv(int argc, char **argv);

  static std::unique_ptr<Context> instance_;
  std::unique_ptr<specfem::MPI::MPI> mpi_;
  bool kokkos_initialized_;
  bool mpi_initialized_;
  bool context_initialized_;
};

/**
 * @brief RAII guard for SPECFEM++ context (similar to Kokkos::ScopeGuard)
 *
 * This class provides automatic initialization and finalization of SPECFEM++
 * resources following the RAII pattern. It initializes Kokkos and MPI in the
 * constructor and ensures proper cleanup in the destructor.
 *
 * Usage:
 * @code
 * int main(int argc, char* argv[]) {
 *     specfem::ContextGuard guard(argc, argv);
 *     // Kokkos & MPI automatically initialized
 *     // ... use context ...
 *     // Automatic cleanup on scope exit
 * }
 * @endcode
 */
class ContextGuard {
public:
  /**
   * @brief Constructor - initializes Context with command line arguments
   *
   * @param argc Command line argument count
   * @param argv Command line arguments
   * @throws std::runtime_error if Context is already initialized or finalized
   */
  ContextGuard(int argc, char *argv[]);

  /**
   * @brief Constructor - initializes Context with argument vector
   *
   * @param args Vector of command line arguments
   * @throws std::runtime_error if Context is already initialized or finalized
   */
  explicit ContextGuard(const std::vector<std::string> &args);

  /**
   * @brief Destructor - automatically finalizes Context
   */
  ~ContextGuard();

  /**
   * @brief Get reference to the Context instance
   *
   * @return Reference to the managed Context
   */
  Context &get_context();

  /**
   * @brief Get MPI instance for convenience
   *
   * @return Pointer to MPI instance
   */
  specfem::MPI::MPI *get_mpi() const;

  /**
   * @brief Delete copy constructor and assignment operator
   *
   * ContextGuard cannot be copied or moved to prevent multiple finalization
   */
  ContextGuard(const ContextGuard &) = delete;
  ContextGuard &operator=(const ContextGuard &) = delete;
  ContextGuard(ContextGuard &&) = delete;
  ContextGuard &operator=(ContextGuard &&) = delete;

private:
  Context &context_;
  bool should_finalize_;
};

} // namespace specfem
