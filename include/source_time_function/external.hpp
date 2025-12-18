#pragma once
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "yaml-cpp/yaml.h"
#include <tuple>
#include <vector>

namespace specfem {
namespace forcing_function {
class external : public stf {
public:
  external(const YAML::Node &external, const int nsteps, const type_real dt);

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) override;

  void update_tshift(type_real tshift) override {
    if (std::abs(tshift) > 1e-6) {
      throw std::runtime_error("Error: external source time function does not "
                               "support time shift");
    }
  }

  std::string print() const override {
    std::stringstream ss;
    ss << "External source time function: "
       << "\n"
       << "  X-component: " << this->x_component_ << "\n"
       << "  Y-component: " << this->y_component_ << "\n"
       << "  Z-component: " << this->z_component_ << "\n";
    return ss.str();
  }

  specfem::enums::seismogram::format get_type() const { return type_; }
  type_real get_t0() const override { return t0_; }
  type_real get_dt() const { return dt_; }
  int get_nsteps() { return nsteps_; }
  int get_ncomponents() const { return ncomponents_; }
  std::string get_x_component() const { return x_component_; }
  std::string get_y_component() const { return y_component_; }
  std::string get_z_component() const { return z_component_; }
  specfem::enums::seismogram::format get_format() const { return type_; }

  bool operator==(const specfem::forcing_function::stf &other) const override;
  bool operator!=(const specfem::forcing_function::stf &other) const override;

private:
  int nsteps_;
  type_real t0_;
  type_real dt_;
  specfem::enums::seismogram::format type_;
  int ncomponents_;
  std::string x_component_ = "";
  std::string y_component_ = "";
  std::string z_component_ = "";
};
} // namespace forcing_function
} // namespace specfem
