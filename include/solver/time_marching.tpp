#pragma once

#include "solver.hpp"
#include "time_marching.hpp"
#include "timescheme/newmark.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::solver::impl {
inline void check_time_marching(const int &istep, const int &nstep,
                                const int &dofs_updated,
                                const int &total_dof_to_be_updated,
                                const int &elements_updated,
                                const int &total_elements_to_be_updated) {
  if (istep % 10 == 0) {
    std::cout << "Progress : executed " << istep << " steps of " << nstep
              << " steps" << std::endl;
  }
  if (dofs_updated != total_dof_to_be_updated) {
    std::ostringstream message;
    message << "The time loop has not updated all the degrees of freedom. "
            << "Only " << dofs_updated << " out of " << total_dof_to_be_updated
            << " degrees of freedom have been updated.";

    throw std::runtime_error(message.str());
  }

  if (elements_updated != total_elements_to_be_updated) {
    std::ostringstream message;
    message << "The time loop has not updated all the elements. "
            << "Only " << elements_updated << " out of "
            << total_elements_to_be_updated << " elements have been updated.";

    throw std::runtime_error(message.str());
  }
}
} // namespace specfem::solver::impl

template <specfem::dimension::type DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::forward,
                                    DimensionTag, NGLL>::run() {
  // Calls to compute mass matrix and invert mass matrix
  this->kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      2 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated =
      assembly.get_total_number_of_elements();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto [istep, dt] : time_scheme->iterate_forward()) {
    int dofs_updated = 0;
    int elements_updated = 0;

    // Predictor phase forward
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC,
                    ELASTIC_PSV_T)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                this->time_scheme->apply_predictor_phase_forward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                this->time_scheme->apply_predictor_phase_forward(_medium_tag_);
          }
        })
    // Update wavefield and apply corrector phase forward
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC,
                    ELASTIC_PSV_T)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                this->kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                this->time_scheme->apply_corrector_phase_forward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                this->kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                this->time_scheme->apply_corrector_phase_forward(_medium_tag_);
          }
        })

    // Compute seismograms if required
    if (time_scheme->compute_seismogram(istep)) {
      this->kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }
    // Run periodic tasks such as plotting, etc.
    for (const auto &task : tasks) {
      if (task && task->should_run(istep + 1)) {
        task->run(assembly, istep + 1);
      }
    }

    specfem::solver::impl::check_time_marching(
        istep, nstep, dofs_updated, total_dof_to_be_updated, elements_updated,
        total_elements_to_be_updated);
  }

  for (const auto &task : tasks) {
    if (task && !task->should_run(nstep) && task->should_run(-1)) {
      task->run(assembly, nstep);
    }
  }

  for (const auto &task : tasks) {
    task->finalize(assembly);
  }

  std::cout << std::endl;

  return;
}

template <specfem::dimension::type DimensionTag, int NGLL>
void specfem::solver::time_marching<specfem::simulation::type::combined,
                                    DimensionTag, NGLL>::run() {
  adjoint_kernels.initialize(time_scheme->get_timestep());
  backward_kernels.initialize(time_scheme->get_timestep());

  const int nstep = time_scheme->get_max_timestep();

  const int total_dof_to_be_updated =
      4 * assembly.get_total_degrees_of_freedom();

  const int total_elements_to_be_updated =
      2 * assembly.get_total_number_of_elements();

  for (const auto &task : tasks) {
    task->initialize(assembly);
  }

  for (const auto &task : tasks) {
    if (task && !task->should_run(nstep) && task->should_run(-1)) {
      task->run(assembly, nstep);
    }
  }

  for (const auto [istep, dt] : time_scheme->iterate_backward()) {
    for (const auto &task : tasks) {
      if (task && task->should_run(istep + 1)) {
        task->run(assembly, istep + 1);
      }
    }

    int dofs_updated = 0;
    int elements_updated = 0;
    // Adjoint time step
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                time_scheme->apply_predictor_phase_forward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                time_scheme->apply_predictor_phase_forward(_medium_tag_);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                adjoint_kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                time_scheme->apply_corrector_phase_forward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                adjoint_kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                time_scheme->apply_corrector_phase_forward(_medium_tag_);
          }
        })

    // Backward time step
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                time_scheme->apply_predictor_phase_backward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            dofs_updated +=
                time_scheme->apply_predictor_phase_backward(_medium_tag_);
          }
        })

    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ACOUSTIC, ELASTIC_PSV, ELASTIC_SH, POROELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                backward_kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                time_scheme->apply_corrector_phase_backward(_medium_tag_);
          }
        })
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC)),
        {
          if constexpr (dimension_tag == _dimension_tag_) {
            elements_updated +=
                backward_kernels.template update_wavefields<_medium_tag_>(istep);
            dofs_updated +=
                time_scheme->apply_corrector_phase_backward(_medium_tag_);
          }
        })

    // Copy read wavefield buffer to the backward wavefield
    // We need to do this after the first backward step to align
    // the wavefields for the adjoint and backward simulations
    // for accurate Frechet derivatives
    if (istep == nstep - 1) {
      specfem::assembly::deep_copy(assembly.fields.backward,
                                   assembly.fields.buffer);
    }

    frechet_kernels.compute_derivatives(dt);

    if (time_scheme->compute_seismogram(istep)) {
      // compute seismogram for backward time step
      backward_kernels.compute_seismograms(time_scheme->get_seismogram_step());
      time_scheme->increment_seismogram_step();
    }

    specfem::solver::impl::check_time_marching(
        istep, nstep, dofs_updated, total_dof_to_be_updated, elements_updated,
        total_elements_to_be_updated);
  }

  for (const auto &task : tasks) {
    task->finalize(assembly);
  }

  std::cout << std::endl;

  return;
}
