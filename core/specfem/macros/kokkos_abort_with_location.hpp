#pragma once

/**
 * @file kokkos_abort_with_location.hpp
 * @brief Macro to abort Kokkos execution with file and line information
 */

#include <Kokkos_Core.hpp>
#include <boost/preprocessor/stringize.hpp>

/**
 * @brief Abort Kokkos execution with a message and location.
 *
 * This macro calls `Kokkos::abort` with a message that includes the file name
 * and line number where the macro is called.
 *
 * @param message The error message to display.
 */
#define KOKKOS_ABORT_WITH_LOCATION(message)                                    \
  Kokkos::abort(__FILE__ ":" BOOST_PP_STRINGIZE(__LINE__) " - " message);
