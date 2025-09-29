#pragma once

#include "absorbing_boundaries/absorbing_boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "enumerations/interface.hpp"
#include "materials/materials.hpp"

namespace specfem::mesh::meshfem3d {

template <specfem::dimension::type DimensionTag> struct mesh;

template <> struct mesh<specfem::dimension::type::dim3> {
  specfem::mesh::meshfem3d::ControlNodes<specfem::dimension::type::dim3>
      control_nodes;
  specfem::mesh::meshfem3d::Materials<specfem::dimension::type::dim3> materials;
  specfem::mesh::meshfem3d::AbsorbingBoundaries<specfem::dimension::type::dim3>
      absorbing_boundaries;

  mesh() = default;
  ~mesh() = default;
};

} // namespace specfem::mesh::meshfem3d
