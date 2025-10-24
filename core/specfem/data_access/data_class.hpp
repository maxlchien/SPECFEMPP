#pragma once

namespace specfem::data_access {
enum DataClassType {
  index,
  edge_index,
  assembly_index,
  gll_index,
  mapped_index,
  properties,
  kernels,
  jacobian_matrix,
  field_derivatives,
  displacement,
  velocity,
  acceleration,
  mass_matrix,
  source,
  stress,
  stress_integrand,
  boundary,
  coupled_interface,
  lagrange_derivative,
  weights
};
} // namespace specfem::data_access
