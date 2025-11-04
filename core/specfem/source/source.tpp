#pragma once

#include "specfem/source.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <yaml-cpp/yaml.h>


template <specfem::dimension::type DimensionTag>
void specfem::sources::source<DimensionTag>::set_forcing_function(
    YAML::Node &Node, const int nsteps, const type_real dt) {

  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function = std::make_unique<specfem::forcing_function::Dirac>(
        Dirac, nsteps, dt, false);
  } else if (YAML::Node Gaussian = Node["Gaussian"]) {
    if (Gaussian["hdur"]) {
      type_real hdur = Gaussian["hdur"].as<type_real>();
      this->forcing_function =
          std::make_unique<specfem::forcing_function::GaussianHdur>(
              nsteps, dt, hdur,
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false);
      return;
    } else if (Gaussian["f0"]) {
      type_real f0 = Gaussian["f0"].as<type_real>();
      this->forcing_function =
          std::make_unique<specfem::forcing_function::Gaussian>(
              nsteps, dt, f0,
              Gaussian["tshift"] ? Gaussian["tshift"].as<type_real>() : 0.0,
              Gaussian["factor"].as<type_real>(), false);
      return;
    } else {
      throw std::runtime_error(
          "Error: Gaussian source time function requires either 'hdur' or 'f0' "
          "to be specified.");
    }
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Ricker>(Ricker, nsteps, dt,
                                                            false);
  } else if (YAML::Node dGaussian = Node["dGaussian"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::dGaussian>(
            dGaussian, nsteps, dt, false);
  } else if (YAML::Node Heaviside = Node["Heaviside"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Heaviside>(
            Heaviside, nsteps, dt, false);
  } else if (YAML::Node external = Node["External"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::external>(external, nsteps,
                                                              dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }

}
