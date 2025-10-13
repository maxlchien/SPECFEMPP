#pragma once

#include "impl/adjacency_graph.hpp"
#include "impl/control_nodes.hpp"
#include "impl/mesh_to_compute_mapping.hpp"
#include "impl/points.hpp"
#include "impl/shape_functions.hpp"
#include "mesh/mesh.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly/mesh/impl/quadrature.hpp"
#include "specfem/element.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include <vector>

namespace specfem::assembly {
/**
 * @brief Information on an assembled mesh
 *
 */
template <>
struct mesh<specfem::dimension::type::dim2>
    : public specfem::assembly::mesh_impl::points<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::quadrature<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::control_nodes<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::mesh_to_compute_mapping<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::shape_functions<
          specfem::dimension::type::dim2>,
      public specfem::assembly::mesh_impl::adjacency_graph<
          specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension
  int nspec;                          ///< Number of spectral
                                      ///< elements
  int ngnod;                          ///< Number of control
                                      ///< nodes
  specfem::mesh_entity::element<dimension_tag> element_grid; ///< Element number
                                                             ///< of GLL points

  mesh() = default;

  mesh(const specfem::mesh::tags<dimension_tag> &tags,
       const specfem::mesh::control_nodes<dimension_tag> &control_nodes,
       const specfem::quadrature::quadratures &quadratures,
       const specfem::mesh::adjacency_graph<dimension_tag> &adjacency_graph);

  // TODO(Rohit: ADJ_GRAPH_DEFAULT)
  // Remove assemble_legacy functionality when adjacency graph is the default
  // feature for store mesh adjancencies
  void assemble_legacy();

  void assemble();

  // TODO(Rohit: ADJ_GRAPH_DEFAULT)
  // The graph should never be empty after it is made as default
  bool adjacency_graph_empty() const {
    return static_cast<const specfem::assembly::mesh_impl::adjacency_graph<
        dimension_tag> &>(*this)
        .empty();
  }

  bool empty() = delete;
};
} // namespace specfem::assembly

#include "data_access.hpp"
