#pragma once

#include "execution/for_all.hpp"
#include "execution/range_iterator.hpp"
#include "parallel_configuration/range_config.hpp"
#include "specfem/timescheme/newmark.hpp"

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::forward>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {


  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<dimension_tag, _medium_tag_, wavefield>(field, deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::forward>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::forward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<dimension_tag, _medium_tag_, wavefield>(
              field, deltat, deltatover2, deltasquareover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::combined>::
    apply_corrector_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<dimension_tag, _medium_tag_, wavefield>(adjoint_field,
                                                    deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::combined>::
    apply_corrector_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       ELASTIC_PSV_T, POROELASTIC)),
      {
        if (tag == _medium_tag_) {
          return corrector_phase_impl<dimension_tag, _medium_tag_, wavefield>(
              backward_field, -1.0 * deltatover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::combined>::
    apply_predictor_phase_forward(const specfem::element::medium_tag tag) {

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::adjoint;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<dimension_tag, _medium_tag_, wavefield>(
              adjoint_field, deltat, deltatover2, deltasquareover2);
        }
      })

  Kokkos::abort("Medium type not supported.");

  /// Code path should never be reached
  return 0;
}

template<>
int specfem::time_scheme::newmark<specfem::assembly::fields<specfem::dimension::type::dim2>,
                                  specfem::simulation::type::combined>::
    apply_predictor_phase_backward(const specfem::element::medium_tag tag) {

  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto wavefield = specfem::wavefield::simulation_field::backward;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2), MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC,
                                       POROELASTIC, ELASTIC_PSV_T)),
      {
        if (tag == _medium_tag_) {
          return predictor_phase_impl<dimension_tag, _medium_tag_, wavefield>(
              backward_field, -1.0 * deltat, -1.0 * deltatover2,
              deltasquareover2);
        }
      })

  Kokkos::abort("Medium type not supported.");
  /// Code path should never be reached
  return 0;
}
