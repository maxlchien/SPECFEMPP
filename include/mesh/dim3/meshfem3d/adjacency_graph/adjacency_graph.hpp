#pragma once

#include "enumerations/interface.hpp"
#include <boost/graph/adjacency_list.hpp>

namespace specfem::mesh::meshfem3d {

template <specfem::dimension::type Dimension> struct adjacency_graph;

template <> struct adjacency_graph<specfem::dimension::type::dim3> {
public:
  struct EdgeProperties {
    specfem::connections::type connection;
    specfem::mesh_entity::dim3::type orientation;

    EdgeProperties() = default;

    EdgeProperties(const specfem::connections::type conn,
                   const specfem::mesh_entity::dim3::type orient)
        : connection(conn), orientation(orient) {}
  };

private:
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
                            boost::no_property, EdgeProperties>;

  Graph graph_;

public:
  adjacency_graph() = default;
  adjacency_graph(const int nspec) : graph_(nspec) {}
  Graph &graph() { return graph_; }
  const Graph &graph() const { return graph_; }
  bool empty() const { return boost::num_vertices(graph_) == 0; }

  void assert_symmetry() const;
};

} // namespace specfem::mesh::meshfem3d
