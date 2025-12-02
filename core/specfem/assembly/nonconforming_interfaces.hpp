#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

namespace nonconforming_interfaces_impl {

template <specfem::dimension::type DimensionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag,
          specfem::connections::type ConnectionTag>
struct interface_container;

} // namespace nonconforming_interfaces_impl

/**
 * @brief Information on coupled interfaces between two mediums
 * @tparam DimensionTag Dimension of spectral elements
 */
template <specfem::dimension::type DimensionTag>
struct nonconforming_interfaces;

} // namespace specfem::assembly

#include "nonconforming_interfaces/dim2/nonconforming_interfaces.hpp"
