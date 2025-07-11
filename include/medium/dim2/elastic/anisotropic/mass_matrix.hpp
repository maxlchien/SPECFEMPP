#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "globals.h"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace medium {

// Using template specializations from isotropic case

// template <bool UseSIMD, specfem::element::property_tag PropertyTag>
// KOKKOS_FUNCTION specfem::point::field<specfem::dimension::type::dim2,
//                                       specfem::element::medium_tag::elastic_psv,
//                                       false, false, false, true, UseSIMD>
// impl_mass_matrix_component(
//     const specfem::point::properties<specfem::dimension::type::dim2,
//                                      specfem::element::medium_tag::elastic_psv,
//                                      PropertyTag, UseSIMD> &properties,
//     const specfem::point::partial_derivatives<
//         specfem::dimension::type::dim2, true, UseSIMD> &partial_derivatives);

} // namespace medium
} // namespace specfem
