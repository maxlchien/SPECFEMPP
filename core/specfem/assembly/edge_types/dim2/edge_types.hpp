#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/macros.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <> class edge_types<specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim2;

private:
  template <typename ExecutionSpace> struct Edge {
    int n_points;
    using IndexView = Kokkos::View<int *, Kokkos::LayoutStride, ExecutionSpace>;
    int element_index;
    int edge_index;
    specfem::mesh_entity::dim2::type edge_type;
    IndexView iz;
    IndexView ix;

    Edge(const int n_points, const int element_index, const int edge_index,
         const specfem::mesh_entity::dim2::type edge_type, const IndexView iz,
         const IndexView ix)
        : n_points(n_points), element_index(element_index),
          edge_index(edge_index), edge_type(edge_type), iz(iz), ix(ix) {}

    KOKKOS_INLINE_FUNCTION
    specfem::point::edge_index<specfem::dimension::type::dim2>
    operator()(const int point_id) const {
      return { element_index, edge_index,   point_id,
               iz(point_id),  ix(point_id), edge_type };
    }
  };

  template <typename ExecutionSpace> struct EdgeView {
    int n_edges;
    int n_points;
    std::string label;
    using IndexView = Kokkos::View<int *, ExecutionSpace>;
    using QPView = Kokkos::View<int **, ExecutionSpace>;
    using EdgeTypeView =
        Kokkos::View<specfem::mesh_entity::dim2::type *, ExecutionSpace>;

    using HostMirror = std::conditional_t<
        std::is_same<ExecutionSpace, Kokkos::HostSpace>::value, EdgeView,
        EdgeView<Kokkos::HostSpace> >;

    EdgeView() : n_edges(0), n_points(0), label("undefined") {}

    EdgeView(const std::string &label, const int n_edges, const int n_points)
        : label(label), n_edges(n_edges), n_points(n_points),
          element_index(label + "_element_index", n_edges),
          edge_index(label + "_edge_index", n_edges),
          edge_types(label + "_edge_types", n_edges),
          iz(label + "_iz", n_edges, n_points),
          ix(label + "_ix", n_edges, n_points) {}

    IndexView element_index;
    IndexView edge_index;
    EdgeTypeView edge_types;
    QPView iz;
    QPView ix;

    EdgeView(const int n_edges, const int n_points, const std::string &label,
             const IndexView &element_index, const IndexView &edge_index,
             const EdgeTypeView &edge_types, const QPView &iz, const QPView &ix)
        : n_edges(n_edges), n_points(n_points), label(label),
          element_index(element_index), edge_index(edge_index),
          edge_types(edge_types), iz(iz), ix(ix) {}

    KOKKOS_INLINE_FUNCTION
    Edge<ExecutionSpace> operator()(const int edge_id) const {
      return { n_points,
               element_index(edge_id),
               edge_index(edge_id),
               edge_types(edge_id),
               Kokkos::subview(iz, edge_id, Kokkos::ALL()),
               Kokkos::subview(ix, edge_id, Kokkos::ALL()) };
    }

    KOKKOS_INLINE_FUNCTION
    EdgeView<ExecutionSpace>
    operator()(const Kokkos::pair<int, int> &edge_range) const {
      return { edge_range.second - edge_range.first,
               n_points,
               label + "_subview",
               Kokkos::subview(element_index, edge_range),
               Kokkos::subview(edge_index, edge_range),
               Kokkos::subview(edge_types, edge_range),
               Kokkos::subview(iz, edge_range, Kokkos::ALL()),
               Kokkos::subview(ix, edge_range, Kokkos::ALL()) };
    }
  };

public:
  using EdgeViewType = EdgeView<Kokkos::DefaultExecutionSpace>;

private:
  static EdgeViewType::HostMirror create_mirror_view(const EdgeViewType &view) {
    return EdgeViewType::HostMirror(view.label, view.n_edges, view.n_points);
  }

  template <typename SrcView, typename DestView>
  static void deep_copy(const DestView &dest, const SrcView &src) {
    Kokkos::deep_copy(dest.element_index, src.element_index);
    Kokkos::deep_copy(dest.edge_index, src.edge_index);
    Kokkos::deep_copy(dest.edge_types, src.edge_types);
    Kokkos::deep_copy(dest.iz, src.iz);
    Kokkos::deep_copy(dest.ix, src.ix);
  }

public:
  std::tuple<typename EdgeViewType::HostMirror,
             typename EdgeViewType::HostMirror>
  get_edges_on_host(const specfem::connections::type connection,
                    const specfem::interface::interface_tag edge,
                    const specfem::element::boundary_tag boundary) const;

  std::tuple<EdgeViewType, EdgeViewType>
  get_edges_on_device(const specfem::connections::type connection,
                      const specfem::interface::interface_tag edge,
                      const specfem::element::boundary_tag boundary) const;

  edge_types(
      const int ngllx, const int ngllz,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::mesh::coupled_interfaces<dimension_tag>
          &coupled_interfaces);

  edge_types() = default;

private:
  FOR_EACH_IN_PRODUCT((DIMENSION_TAG(DIM2),
                       CONNECTION_TAG(WEAKLY_CONFORMING, NONCONFORMING),
                       INTERFACE_TAG(ELASTIC_ACOUSTIC, ACOUSTIC_ELASTIC),
                       BOUNDARY_TAG(NONE, ACOUSTIC_FREE_SURFACE, STACEY,
                                    COMPOSITE_STACEY_DIRICHLET)),
                      DECLARE((EdgeViewType, self_edges),
                              (EdgeViewType::HostMirror, h_self_edges),
                              (EdgeViewType, coupled_edges),
                              (EdgeViewType::HostMirror, h_coupled_edges)))
};

} // namespace specfem::assembly
