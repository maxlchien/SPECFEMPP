#include "algorithms/locate_point.hpp"
#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
#include "enumerations/specfem_enums.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/point.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <cmath>

void specfem::sources::force::compute_source_array(
    const specfem::compute::mesh &mesh,
    const specfem::compute::partial_derivatives &partial_derivatives,
    const specfem::compute::element_types &element_types,
    specfem::kokkos::HostView3d<type_real> source_array) {

  specfem::point::global_coordinates<specfem::dimension::type::dim2> coord(
      this->x, this->z);
  auto lcoord = specfem::algorithms::locate_point(coord, mesh);

  const auto xi = mesh.quadratures.gll.h_xi;
  const auto gamma = mesh.quadratures.gll.h_xi;
  const auto N = mesh.quadratures.gll.N;

  const auto el_type = element_types.get_medium_tag(lcoord.ispec);
  const int ncomponents = source_array.extent(0);

  // Compute lagrange interpolants at the source location
  auto [hxi_source, hpxi_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.xi, N, xi);
  auto [hgamma_source, hpgamma_source] =
      specfem::quadrature::gll::Lagrange::compute_lagrange_interpolants(
          lcoord.gamma, N, gamma);

  type_real hlagrange;

  // Source array computation
  for (int iz = 0; iz < N; ++iz) {
    for (int ix = 0; ix < N; ++ix) {
      hlagrange = hxi_source(ix) * hgamma_source(iz);

      // Acoustic
      if (el_type == specfem::element::medium_tag::acoustic) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Force source requires 1 component for acoustic medium");
        }
        source_array(0, iz, ix) = hlagrange;
      }
      // Elastic SH
      else if (el_type == specfem::element::medium_tag::elastic_sh) {
        if (ncomponents != 1) {
          throw std::runtime_error(
              "Force source requires 1 component for elastic SH medium");
        }
        source_array(0, iz, ix) = hlagrange;

      } else if ((el_type == specfem::element::medium_tag::elastic_psv)) {
        if (ncomponents != 2) {
          throw std::runtime_error(
              "Force source requires 2 components for elastic, "
              "poroelastic, or electromagnetic-sv media.");
        }
        source_array(0, iz, ix) =
            std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(1, iz, ix) =
            -1.0 *
            std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
      } else if ((el_type == specfem::element::medium_tag::poroelastic)) {
        if (ncomponents != 4) {
          throw std::runtime_error(
              "Force source requires 4 components for poroelastic medium");
        }
        source_array(0, iz, ix) =
            std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(1, iz, ix) =
            -1.0 *
            std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(2, iz, ix) =
            std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(3, iz, ix) =
            -1.0 *
            std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
      } else if (el_type == specfem::element::medium_tag::elastic_psv_t) {
        if (ncomponents != 3) {
          throw std::runtime_error(
              "Force source requires 3 components for elastic psv_t medium");
        }
        source_array(0, iz, ix) =
            std::sin(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(1, iz, ix) =
            -1.0 *
            std::cos(Kokkos::numbers::pi_v<type_real> / 180 * this->angle) *
            hlagrange;
        source_array(2, iz, ix) = static_cast<type_real>(0.0);
      } else {
        std::ostringstream message;
        message << "Source array computation not implemented for element type: "
                << specfem::element::to_string(el_type) << " [" << __FILE__
                << ":" << __LINE__ << "]";
        auto message_str = message.str();
        Kokkos::abort(message_str.c_str());
      }
    }
  }
}

std::string specfem::sources::force::print() const {

  std::ostringstream message;
  message << "- Force Source: \n"
          << "    Source Location: \n"
          << "      x = " << type_real(this->x) << "\n"
          << "      z = " << type_real(this->z) << "\n"
          << "    Source Time Function: \n"
          << this->forcing_function->print() << "\n";

  return message.str();
}

bool almost_equal(const type_real a, const type_real b,
                  const type_real epsilon = 1e-6) {
  return std::abs(a - b) < epsilon;
}

bool specfem::sources::force::operator==(
    const specfem::sources::source &other) const {

  // Try casting the other source to a force source
  const auto *other_source =
      dynamic_cast<const specfem::sources::force *>(&other);

  // Check if cast was successful
  if (other_source == nullptr) {
    std::cout << "Other source is not a force object" << std::endl;
    return false;
  }

  return specfem::utilities::almost_equal(this->x, other_source->x) &&
         specfem::utilities::almost_equal(this->z, other_source->z) &&
         specfem::utilities::almost_equal(this->angle, other_source->angle) &&
         *(this->forcing_function) == *(other_source->forcing_function);
}
bool specfem::sources::force::operator!=(
    const specfem::sources::source &other) const {
  return !(*this == other);
}
