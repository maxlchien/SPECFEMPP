#pragma once

#include <cstdlib>
#include <iostream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem {

// Forward declaration
namespace program {
class Context;
}

/**
 * @class MPI
 * @brief Static MPI wrapper for SPECFEM++
 *
 * This class provides a static interface to MPI functionality,
 * eliminating the need to pass MPI pointers throughout the codebase.
 *
 * Key features:
 * - Static rank and size members accessible globally
 * - Context-managed lifecycle (only Context can initialize/finalize)
 * - Safety checks to prevent use outside Context scope
 * - Minimal API focused on essential MPI operations
 *
 * Usage:
 * @code
 * // After Context is initialized
 * int my_rank = specfem::MPI::MPI::rank;
 * int world_size = specfem::MPI::MPI::size;
 * specfem::MPI::MPI::sync();
 * @endcode
 *
 * @note This class cannot be instantiated. All members are static.
 * @note Only specfem::program::Context can initialize/finalize this class.
 */
class MPI_new {
public:
  static int rank; ///< Current MPI rank (-1 if not initialized)
  static int size; ///< Total number of MPI processes (-1 if not initialized)

  /**
   * @brief Synchronize all MPI processes (MPI_Barrier)
   *
   * @throws Exits with error code 1 if called outside Context scope
   */
  static void sync() {
    check_context();
#ifdef MPI_PARALLEL
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }

  /**
   * @brief Print strings from the head node
   *
   */

  template <typename T> static void cout(T s, bool root_only = true) {
#ifdef MPI_PARALLEL
    if (rank == 0 || !root_only) {
      std::cout << s << std::endl;
    }
#else
    std::cout << s << std::endl;
#endif
  }

  template <typename T> static void print(T s) { cout(s, true); }

private:
  MPI_new() = default;
  ~MPI_new() = default;
  MPI_new(const MPI_new &) = delete;
  MPI_new &operator=(const MPI_new &) = delete;

  /**
   * @brief Initialize MPI and set rank/size
   *
   * Called by Context constructor. Checks if MPI is already initialized
   * externally before calling MPI_Init.
   *
   * @param argc Pointer to argument count
   * @param argv Pointer to argument vector
   */
  static void initialize(int *argc, char ***argv);

  /**
   * @brief Finalize MPI and reset rank/size to -1
   *
   * Called by Context destructor. Only calls MPI_Finalize if MPI
   * was initialized by this wrapper (not externally).
   */
  static void finalize();

  /**
   * @brief Check if MPI is initialized (Context exists)
   *
   * Verifies that rank and size are valid (not -1).
   * Exits with error code 1 if check fails.
   */
  static void check_context() {
    if (rank == -1 || size == -1) {
      std::cerr << "ERROR: MPI used outside Context scope" << std::endl;
      std::exit(1);
    }
  }

  friend class specfem::program::Context;
};

} // namespace specfem
