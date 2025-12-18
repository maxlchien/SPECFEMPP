#pragma once

/**
 * @brief Namespace for source time functions
 *
 * @details This namespace contains various source time functions (STFs). Each
 * STF class inherits from the base class specfem::forcing_function::stf and
 * implements specific time-dependent behavior for seismic sources. The STFs
 * defined here can be used to model different types of seismic source time.
 * The available source time functions include:
 * - Dirac: Represents an instantaneous impulse.
 * - Gaussian: Models a Gaussian-shaped pulse.
 * - Ricker: Implements the Ricker wavelet, commonly used in seismology.
 * - dGaussian: Represents the derivative of a Gaussian pulse.
 * - Heaviside: Models a step function.
 * - external: Allows for user-defined source time functions.
 *
 * @see specfem::sources::source::set_forcing_function for how to the forcing
 * function is set up from the source class and YAML configuration.
 */
namespace specfem::forcing_function {}

#include "specfem/source_time_functions/dgaussian.hpp"
#include "specfem/source_time_functions/dirac.hpp"
#include "specfem/source_time_functions/external.hpp"
#include "specfem/source_time_functions/gaussian.hpp"
#include "specfem/source_time_functions/gaussianhdur.hpp"
#include "specfem/source_time_functions/heaviside.hpp"
#include "specfem/source_time_functions/ricker.hpp"
#include "specfem/source_time_functions/source_time_function.hpp"
