#ifndef _RECEIVER_HPP
#define _RECEIVER_HPP

#include "compute/compute_mesh.hpp"
#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

namespace specfem {
namespace receivers {

/**
 * @brief Receiver Class
 *
 */
class receiver {

public:
  /**
   * @brief Construct a new receiver object
   *
   * @param network_name Name of network where this station lies in
   * @param station_name Name of station
   * @param x X coordinate of the station
   * @param z Z coordinate of the station
   * @param angle Angle of the station
   */
  receiver(const std::string network_name, const std::string station_name,
           const type_real x, const type_real z, const type_real angle)
      : network_name(network_name), station_name(station_name), x(x), z(z),
        angle(angle) {};
  /**
   * @brief Compute the receiver array (lagrangians) for this station
   *
   * @param quadx Quadrature object in x-dimension
   * @param quadz Quadrature object in z-dimension
   * @param receiver_array view to store the source array
   */
  void
  compute_receiver_array(const specfem::compute::mesh &mesh,
                         specfem::kokkos::HostView3d<type_real> receiver_array);
  /**
   * @brief Get the name of network where this station lies
   *
   * @return std::string name of the network where the station lies
   */
  std::string get_network_name() { return this->network_name; }
  /**
   * @brief Get the name of this station
   *
   * @return std::string Name of this station
   */
  std::string get_station_name() { return this->station_name; }

  /**
   * @brief User output
   *
   */
  std::string print() const;

  type_real get_angle() const { return this->angle; }
  type_real get_x() const { return this->x; }
  type_real get_z() const { return this->z; }

private:
  type_real x;              ///< x coordinate of source
  type_real z;              ///< z coordinate of source
  type_real angle;          ///< Angle to rotate components at receivers
  std::string network_name; ///< Name of the network where this station lies
  std::string station_name; ///< Name of the station
};
} // namespace receivers

} // namespace specfem

#endif
