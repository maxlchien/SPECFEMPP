#pragma once

namespace specfem {
namespace compute {
class assembly;
}
} // namespace specfem

namespace specfem {
namespace io {
/**
 * @brief Base reader class
 *
 */
class reader {
public:
  /**
   * @brief Method to execute the read operation
   *
   * @param assembly Assembly object
   *
   */
  virtual void read(specfem::compute::assembly &assembly) = 0;
};
} // namespace io
} // namespace specfem
