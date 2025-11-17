#pragma once

#include "macros/enum_tags.hpp"
#include "macros/kokkos_abort_with_location.hpp"
#include "macros/material_definitions.hpp"
#include "macros/suppress_warnings.hpp"

#ifndef NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)
#else // NDEBUG
#define ASSERT(condition, message)                                             \
  do {                                                                         \
  } while (false)
#endif
