#pragma once

#include <source_location>
#include <string>

namespace specfem::program {

/**
 * @brief Abort the program with proper MPI cleanup
 *
 * Checks if the program context is initialized (via MPI_new state) and
 * calls MPI_Abort if MPI is active, otherwise calls std::exit.
 * Logs the error message using Logger if context exists, otherwise prints
 * to std::cerr.
 *
 * @param message Error message to display (default: empty string)
 * @param error_code Error code to return (default: 30)
 * @param source_loc Source location information (default: current location)
 *
 * @note This function does not return - the program will be terminated
 */
void abort(const std::string &message = "", int error_code = 30,
           const int line = -1, const char *file = "");

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define specfem_abort(msg, code)                                               \
  specfem::program::abort(msg, code, __LINE__, __FILE__)
} // namespace specfem::program
