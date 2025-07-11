#include "source_time_function/external.hpp"
#include "enumerations/specfem_enums.hpp"
#include "io/seismogram/reader.hpp"
#include "kokkos_abstractions.h"
#include "utilities/strings.hpp"
#include <fstream>
#include <tuple>
#include <vector>

specfem::forcing_function::external::external(const YAML::Node &external,
                                              const int nsteps,
                                              const type_real dt)
    : __nsteps(nsteps), __dt(dt) {

  if (specfem::utilities::is_ascii_string(
          external["format"].as<std::string>()) ||
      !external["format"]) {
    this->type = specfem::enums::seismogram::format::ascii;
  } else {
    throw std::runtime_error("Only ASCII format is supported");
  }

  // Get the components from the file
  // Atleast one component is required
  if (const YAML::Node &stf = external["stf"]) {
    if (stf["X-component"] || stf["Z-component"]) {
      this->x_component =
          (stf["X-component"]) ? stf["X-component"].as<std::string>() : "";
      this->z_component =
          (stf["Z-component"]) ? stf["Z-component"].as<std::string>() : "";
      this->ncomponents = 2;
    } else if (stf["Y-component"]) {
      this->y_component = stf["Y-component"].as<std::string>();
      this->ncomponents = 1;
    } else {
      throw std::runtime_error("Error: External source time function requires "
                               "at least one component");
    }
  } else {
    throw std::runtime_error("Error: External source time function requires "
                             "at least one component");
  }

  // Get t0 and dt from the file
  const std::string filename = [&]() -> std::string {
    if (this->ncomponents == 2) {
      if (this->x_component.empty()) {
        return this->z_component;
      } else {
        return this->x_component;
      }
    } else {
      return this->y_component;
    }
  }();

  std::ifstream file(filename);
  if (!file.good()) {
    throw std::runtime_error("Error: External source time function file " +
                             filename + " does not exist");
  }

  std::string line;
  std::getline(file, line);
  std::istringstream iss(line);
  type_real time, value;
  if (!(iss >> time >> value)) {
    throw std::runtime_error("Seismogram file " + filename +
                             " is not formatted correctly");
  }
  this->__t0 = time;

  std::getline(file, line);
  std::istringstream iss2(line);
  type_real time2, value2;
  iss2 >> time2 >> value2;
  this->__dt = time2 - time;
  file.close();

  return;
}

void specfem::forcing_function::external::compute_source_time_function(
    const type_real t0, const type_real dt, const int nsteps,
    specfem::kokkos::HostView2d<type_real> source_time_function) {

  const int ncomponents = source_time_function.extent(1);

  if (ncomponents != 2) {
    throw std::runtime_error("External source time function only supports 2 "
                             "components");
  }

  if (std::abs(t0 - this->__t0) > 1e-6) {
    throw std::runtime_error(
        "The start time of the external source time "
        "function does not match the simulation start time");
  }

  if (std::abs(dt - this->__dt) > 1e-6) {
    throw std::runtime_error(
        "The time step of the external source time "
        "function does not match the simulation time step");
  }

  std::vector<std::string> filename =
      (ncomponents == 2)
          ? std::vector<std::string>{ this->x_component, this->z_component }
          : std::vector<std::string>{ this->y_component };

  // Check if files exist
  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    // Skip empty filenames
    if (filename[icomp].empty())
      continue;

    std::ifstream file(filename[icomp]);
    if (!file.good()) {
      throw std::runtime_error("Error: External source time function file " +
                               filename[icomp] + " does not exist");
    }
  }

  // set source time function to 0
  for (int i = 0; i < nsteps; i++) {
    for (int icomp = 0; icomp < ncomponents; ++icomp) {
      source_time_function(i, icomp) = 0.0;
    }
  }

  for (int icomp = 0; icomp < ncomponents; ++icomp) {
    if (filename[icomp].empty())
      continue;

    specfem::kokkos::HostView2d<type_real> data("external", nsteps, 2);
    specfem::io::seismogram_reader reader(
        filename[icomp], specfem::enums::seismogram::format::ascii, data);
    reader.read();
    for (int i = 0; i < nsteps; i++) {
      source_time_function(i, icomp) = data(i, 1);
    }
  }
  return;
}

bool specfem::forcing_function::external::operator==(
    const specfem::forcing_function::stf &other) const {
  // First check base class equality
  if (!specfem::forcing_function::stf::operator==(other))
    return false;

  // Then check if the other object is a dGaussian
  auto other_external =
      dynamic_cast<const specfem::forcing_function::external *>(&other);
  if (!other_external)
    return false;

  return (this->x_component == other_external->x_component &&
          this->y_component == other_external->y_component &&
          this->z_component == other_external->z_component &&
          this->__t0 == other_external->__t0 &&
          this->__dt == other_external->__dt &&
          this->type == other_external->type &&
          this->ncomponents == other_external->ncomponents &&
          this->__nsteps == other_external->__nsteps);
};

bool specfem::forcing_function::external::operator!=(
    const specfem::forcing_function::stf &other) const {
  return !(*this == other);
}
