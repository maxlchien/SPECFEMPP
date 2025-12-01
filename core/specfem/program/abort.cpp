#include "specfem/program/abort.hpp"
#include "specfem/logger.hpp"
#include "specfem/mpi.hpp"
#include <iostream>

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace specfem::program {

void abort(const std::string &message, int error_code, const int line,
           const char *file) {

  // convert file and line to string for logging
  std::string filename = std::string(file);
  std::string lineno = std::to_string(line);
  std::string full_message;

  if (!filename.empty() && !lineno.empty() && line != -1) {
    // prepend location info to message
    full_message = filename + ":" + lineno + ": " + message;
  } else {
    full_message = message;
  }

  // Print/log error message if provided
  if (!full_message.empty()) {
    // Check if we're inside a valid program context
    if (specfem::MPI_new::rank != -1 && specfem::MPI_new::size != -1) {
      // Context exists, use Logger
      try {
        specfem::Logger::error(full_message, false); // Print on all ranks
      } catch (...) {
        // If Logger fails, fall back to cerr
        std::cerr << "ERROR: " << full_message << std::endl;
      }
    } else {
      // No context, use cerr
      std::cerr << "ERROR: " << full_message << std::endl;
    }
  }

  // Check if we're inside a valid program context by checking MPI_new state
  if (specfem::MPI_new::rank != -1 && specfem::MPI_new::size != -1) {
    // MPI is initialized, use MPI_Abort for proper parallel cleanup
#ifdef MPI_PARALLEL
    MPI_Abort(MPI_COMM_WORLD, error_code);
#else
    std::exit(error_code);
#endif
  } else {
    // No valid context, just exit
    std::exit(error_code);
  }
}

} // namespace specfem::program
