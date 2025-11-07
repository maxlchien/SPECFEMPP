#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace specfem {
namespace execution {

/**
 * @brief Execution model enumeration
 *
 * Defines the different execution models supported by SPECFEM++.
 * Currently supports 2D and 3D simulations, with room for future
 * expansion (e.g., globe simulations).
 */
enum class model { dim2, dim3 };

/**
 * @brief Convert string to execution model
 *
 * @param str String representation of execution model
 * @return execution::model Corresponding execution model
 * @throws std::invalid_argument if string is not a valid execution model
 */
inline model from_string(const std::string &str) {
  static const std::unordered_map<std::string, execution::model> executions = {
    { "2d", execution::model::dim2 },
    { "dim2", execution::model::dim2 },
    { "3d", execution::model::dim3 },
    { "dim3", execution::model::dim3 }
  };

  auto it = executions.find(str);
  if (it == executions.end()) {
    throw std::invalid_argument("Invalid execution model: " + str +
                                ". Use '2d', 'dim2', '3d', or 'dim3'.");
  }
  return it->second;
}

/**
 * @brief Convert execution model to string
 *
 * @param exec Execution model
 * @return std::string String representation
 */
inline std::string to_string(execution::model exec) {
  switch (exec) {
  case execution::model::dim2:
    return "2D";
  case execution::model::dim3:
    return "3D";
  default:
    return "Unknown";
  }
}

} // namespace execution
} // namespace specfem
