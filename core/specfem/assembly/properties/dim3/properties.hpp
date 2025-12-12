#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "medium/material.hpp"
#include "medium/properties_container.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <vector>

namespace specfem::assembly {

template <>
struct properties<specfem::dimension::type::dim3>
    : public impl::value_containers<specfem::dimension::type::dim3,
                                    specfem::medium::properties_container> {
  /**
   * @name Constructors
   */
  ///@{

  /**
   * @brief Default constructor
   *
   */
  properties() = default;

  properties(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::materials<dimension_tag> &materials,
      const specfem::assembly::element_types<dimension_tag> &element_types);

  ///@}

  int nspec; ///< total number of spectral elements
  int ngllz; ///< number of quadrature points in z dimension
  int nglly; ///< number of quadrature points in y dimension
  int ngllx; ///< number of quadrature points in x dimension

  /**
   * @brief Copy misfit kernel data to host
   *
   */
  void copy_to_host() {
    impl::value_containers<
        dimension_tag, specfem::medium::properties_container>::copy_to_host();
  }

  void copy_to_device() {
    impl::value_containers<
        dimension_tag, specfem::medium::properties_container>::copy_to_device();
  }
};
} // namespace specfem::assembly
