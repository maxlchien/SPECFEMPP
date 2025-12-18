#pragma once
#include "kokkos_abstractions.h"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {

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
 * - External: Allows for user-defined source time functions.
 *
 * @see specfem::source::source::set_forcing_function for how to the forcing
 * function is set up from the source class and YAML configuration.
 */
namespace forcing_function {

/**
 * @note The STF function should lie on host and device. Decorate the class
 * functions within using KOKKOS_FUNCTION macro
 *
 */

/**
 * @brief Source time function base class
 *
 */
class stf {
public:
  /**
   * @brief Default constructor
   *
   */
  stf() {};
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  virtual void update_tshift(type_real tshift) {};
  /**
   * @brief
   *
   */
  virtual type_real get_t0() const { return 0.0; }

  virtual type_real get_tshift() const { return 0.0; }

  virtual std::string print() const = 0;

  virtual bool operator==(const stf &other) const {
    // Base implementation might just check type identity
    return typeid(*this) == typeid(other);
  }
  virtual bool operator!=(const specfem::forcing_function::stf &other) const {
    return !(*this == other);
  }

  // virtual void print(std::ostream &out) const;

  virtual ~stf() = default;

  virtual void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) = 0;
};
} // namespace forcing_function
} // namespace specfem
