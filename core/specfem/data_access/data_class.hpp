#pragma once

namespace specfem::data_access {
enum DataClassType {
  index,
  edge_index,
  assembly_index,
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
  lagrange_derivative,
  weights,
  transfer_function_self,
  transfer_function_coupled,
  intersection_factor,
  intersection_normal,
  nonconforming_interface,
  conforming_interface
};
} // namespace specfem::data_access
