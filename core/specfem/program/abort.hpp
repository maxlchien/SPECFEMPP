#pragma once

#include <string>

namespace specfem::program {

// TODO (LUCAS : CPP20 updated) : use source_location when moving to C++20

/**
 * @brief Terminate program with proper MPI cleanup
 *
 * Calls MPI_Abort if MPI is active, otherwise std::exit. Logs error using
 * Logger if context exists, otherwise prints to std::cerr.
 *
 * @param message Error message to display
 * @param error_code Exit code (default: 30)
 * @param line Source line number (default: -1)
 * @param file Source file name (default: "")
 */
[[noreturn]]
void abort(const std::string &message = "", int error_code = 30,
           const int line = -1, const char *file = "");

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define specfem_abort(msg, code)                                               \
  specfem::program::abort(msg, code, __LINE__, __FILE__)
} // namespace specfem::program
