#pragma once

#include "kokkos_abstractions.h"
#include "source_time_function.hpp"
#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <ostream>

namespace specfem {
namespace forcing_function {
class GaussianHdur : public stf {

public:
  /**
   * @brief Construct a Gaussian source time function object
   *
   * @param hdur half duration
   * @param tshift tshift value
   * @param factor factor to scale source time function
   * @param use_trick_for_better_pressure
   */
  GaussianHdur(const int nsteps, const type_real dt, const type_real hdur,
               const type_real tshift, const type_real factor,
               const bool use_trick_for_better_pressure);

  GaussianHdur(YAML::Node &GaussianNode, const int nsteps, const type_real dt,
               const bool use_trick_for_better_pressure);

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

  std::string print() const override;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) override;

  type_real get_dt() const { return this->dt_; }
  type_real get_factor() const { return this->factor_; }
  type_real get_hdur() const { return this->hdur_; }
  int get_nsteps() const { return this->nsteps_; }
  bool get_use_trick_for_better_pressure() const {
    return this->use_trick_for_better_pressure_;
  }
  int get_ncomponents() const { return 1; }

  bool operator==(const specfem::forcing_function::stf &other) const override;
  bool operator!=(const specfem::forcing_function::stf &other) const override;

private:
  int nsteps_;                         /// number of time steps
  type_real hdur_;                     ///< half duration
  type_real tshift_;                   ///< value of tshift
  type_real t0_;                       ///< t0 value
  type_real t0_factor_;                ///< for the start time computation
  type_real factor_;                   ///< scaling factor
  bool use_trick_for_better_pressure_; /// flag to use trick for better pressure
  type_real dt_;                       ///< time step size
};

} // namespace forcing_function
} // namespace specfem
