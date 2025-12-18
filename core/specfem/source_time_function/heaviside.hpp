#pragma once

#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {

/**
 * @brief Ricker source time function. The Ricker wavelet is a commonly used
 *        source time function in seismic modeling. We define it here as the
 * first derivative of a Ga
 */
class Heaviside : public stf {

public:
  /**
   * @brief Contruct a Heaviside source time function object
   *
   * @param hdur frequency f0
   * @param tshift tshift value
   * @param factor factor to scale source time function
   * @param use_trick_for_better_pressure
   */
  Heaviside(const int nsteps, const type_real dt, const type_real hdur,
            const type_real tshift, const type_real factor,
            const bool use_trick_for_better_pressure,
            const type_real t0_factor = 2.0);

  /**
   * @brief Construct a new Heaviside object
   *
   * @param HeavisideNode
   * @param nsteps
   * @param dt
   * @param use_trick_for_better_pressure
   */
  Heaviside(YAML::Node &HeavisideNode, const int nsteps, const type_real dt,
            const bool use_trick_for_better_pressure,
            const type_real t0_factor = 2.0);

  /**
   * @brief compute the value of stf at time t
   *
   * @param t
   * @return value of source time function at time t
   */
  type_real compute(type_real t);
  /**
   * @brief update the time shift value
   *
   * @param tshift new tshift value
   */
  void update_tshift(type_real tshift) override { this->tshift_ = tshift; }
  /**
   * @brief Get the t0 value
   *
   * @return t0 value
   */
  type_real get_t0() const override { return this->t0_; }

  type_real get_tshift() const override { return this->tshift_; }

  type_real get_dt() const { return this->dt_; }

  type_real get_factor() const { return this->factor_; }

  type_real get_hdur() const { return this->hdur_; }
  int get_nsteps() const { return this->nsteps_; }
  bool get_use_trick_for_better_pressure() const {
    return this->use_trick_for_better_pressure_;
  }
  int get_ncomponents() const { return 1; }

  std::string print() const override;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) override;

  bool operator==(const specfem::forcing_function::stf &other) const override;
  bool operator!=(const specfem::forcing_function::stf &other) const override;

private:
  int nsteps_;          /// number of time steps
  type_real hdur_;      ///< Half duration
  type_real tshift_;    ///< value of tshift
  type_real t0_;        ///< t0 value
  type_real t0_factor_; ///< precomputed factor for t0 calculation
  type_real factor_;    ///< scaling factor
  bool use_trick_for_better_pressure_;
  type_real dt_;
};

} // namespace forcing_function
} // namespace specfem
