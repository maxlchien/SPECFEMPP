#pragma once

#include "enumerations/interface.hpp"
#include "specfem/periodic_tasks/periodic_task.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace specfem::program {

/**
 * @brief RAII-based context managing Kokkos and MPI lifecycle
 *
 * Initializes Kokkos and MPI on construction, finalizes on destruction.
 * Non-copyable and non-movable to ensure single initialization/finalization.
 */
class Context {
public:
  /**
   * @brief Initialize context from command line arguments
   * @param argc Argument count
   * @param argv Argument vector
   */
  Context(int argc, char *argv[]);

  /**
   * @brief Initialize context from argument vector
   * @param args Vector of arguments
   */
  explicit Context(const std::vector<std::string> &args);

  /**
   * @brief Finalize Kokkos and MPI
   */
  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&) = delete;
  Context &operator=(Context &&) = delete;

private:
  static void setup_argc_argv(const std::vector<std::string> &args, int &argc,
                              char **&argv);
  static void cleanup_argc_argv(int argc, char **argv);

  std::unique_ptr<Kokkos::ScopeGuard> kokkos_guard_;
};

} // namespace specfem::program
