#pragma once
#include <string>

namespace specfem {
namespace simulation {

/**
 * @brief Simulation execution types.
 */
enum class type {
  forward, ///< Forward simulation only
  combined ///< Combined forward and adjoint simulation
};

/**
 * @brief Simulation type traits template.
 * @tparam SimulationType Simulation type (forward or combined)
 */
template <specfem::simulation::type SimulationType> class simulation;

/**
 * @brief Forward simulation type traits.
 */
template <> class simulation<specfem::simulation::type::forward> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::forward;

  /**
   * @brief Get simulation type name.
   * @return String representation of simulation type
   */
  static std::string to_string() { return "Forward"; }
};

/**
 * @brief Combined simulation type traits.
 */
template <> class simulation<specfem::simulation::type::combined> {
public:
  static constexpr auto simulation_type = specfem::simulation::type::combined;

  /**
   * @brief Get simulation type name.
   * @return String representation of simulation type
   */
  static std::string to_string() { return "Adjoint & Forward combined"; }
};

} // namespace simulation
} // namespace specfem
