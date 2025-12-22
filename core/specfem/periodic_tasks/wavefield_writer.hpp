#pragma once
#include "enumerations/interface.hpp"
#include "io/operators.hpp"
#include "io/wavefield/writer.hpp"
#include "periodic_task.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace periodic_tasks {
/**
 * @brief Periodic task to write wavefield data during simulation
 *
 * @tparam IOLibrary Template for the I/O library to use for writing
 */
template <template <typename OpType> class IOLibrary>
class wavefield_writer : public periodic_task {
private:
  specfem::io::wavefield_writer<IOLibrary<specfem::io::write> > writer;

public:
  wavefield_writer(const std::string &output_folder, const int time_interval,
                   const bool include_last_step,
                   const bool save_boundary_values)
      : periodic_task(time_interval, include_last_step),
        writer(specfem::io::wavefield_writer<IOLibrary<specfem::io::write> >(
            output_folder, save_boundary_values)) {}

  /**
   * @brief Write wavefield data to file
   *
   */
  void
  run(specfem::assembly::assembly<specfem::dimension::type::dim2> &assembly,
      const int istep) override {
    std::cout << "Writing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.run(assembly, istep);
  }

  /**
   * @brief Write coordinates of wavefield data to disk.
   */
  void initialize(specfem::assembly::assembly<specfem::dimension::type::dim2>
                      &assembly) override {
    std::cout << "Writing coordinate files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.initialize(assembly);
  }

  void finalize(specfem::assembly::assembly<specfem::dimension::type::dim2>
                    &assembly) override {
    std::cout << "Finalizing wavefield files:" << std::endl;
    std::cout << "-------------------------------" << std::endl;
    writer.finalize(assembly);
  }
};

} // namespace periodic_tasks
} // namespace specfem
