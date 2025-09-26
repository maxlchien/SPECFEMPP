#pragma once

#include "enumerations/coupled_interface.hpp"
#include "enumerations/interface.hpp"
#include "enumerations/macros.hpp"
#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/coupled_interfaces.hpp"
#include "specfem/assembly/edge_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.tpp"
#include "specfem/data_access.hpp"

template <specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
specfem::assembly::nonconforming_interfaces_impl::interface_container<
    specfem::dimension::type::dim2, InterfaceTag, BoundaryTag>::
    interface_container(
        const int ngllz, const int ngllx,
        const specfem::assembly::edge_types<specfem::dimension::type::dim2>
            &edge_types,
        const specfem::assembly::jacobian_matrix<dimension_tag>
            &jacobian_matrix,
        const specfem::assembly::mesh<dimension_tag> &mesh) {

  // TODO: make this a parameter for now, use same gll quadrature
  Kokkos::View<type_real*, Kokkos::HostSpace> interface_quadrature(
      "interface_quadrature", ngllx);
  for (int i = 0; i < mesh.h_xi.extent(0); i++) {
    interface_quadrature(i) = mesh.h_xi(i);
  }

  const int nquad_interface = interface_quadrature.extent(0);

  // TODO: replace with settings derived from flux scheme
  bool store_edge_normal =
      (InterfaceTag == specfem::interface::interface_tag::elastic_acoustic);

  // transfer function setting is not symmetric, so we need to make sure we know
  // what side 1 is.
  bool is_side1 =
      (InterfaceTag == specfem::interface::interface_tag::elastic_acoustic);

  if (ngllz <= 0 || ngllx <= 0) {
    KOKKOS_ABORT_WITH_LOCATION("Invalid GLL grid size");
  }

  if (ngllz != ngllx) {
    KOKKOS_ABORT_WITH_LOCATION(
        "The number of GLL points in z and x must be the same.");
  }

  const auto connection_mapping =
      specfem::connections::connection_mapping(ngllx, ngllz);

  const auto [self_edges, coupled_edges] = edge_types.get_edges_on_host(
      specfem::connections::type::nonconforming, InterfaceTag, BoundaryTag);

  const auto nedges = self_edges.size();

  this->edge_factor =
      EdgeFactorView("specfem::assembly::nonconforming_interfaces::edge_factor",
                     nedges, ngllx);

  if (store_edge_normal) {
    this->edge_normal = EdgeNormalView(
        "specfem::assembly::nonconforming_interfaces::edge_normal", nedges,
        nquad_interface, 2);
    this->h_edge_normal = Kokkos::create_mirror_view(edge_normal);
  }

  // consider linking conjugate containers so that we don't need to do
  // set_transfer_functions twice.
  this->transfer_function = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function", nedges,
      nquad_interface, ngllx);

  this->transfer_function_other = TransferFunctionView(
      "specfem::assembly::nonconforming_interfaces::transfer_function_other",
      nedges, nquad_interface, ngllx);

  this->h_edge_factor = Kokkos::create_mirror_view(edge_factor);
  this->h_transfer_function = Kokkos::create_mirror_view(transfer_function);
  this->h_transfer_function_other = Kokkos::create_mirror_view(transfer_function_other);

  const auto weights = mesh.h_weights;

  // used when computing transfer functions
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      icoorg("icoorg", mesh.ngnod);
  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim2> *,
      Kokkos::HostSpace>
      jcoorg("jcoorg", mesh.ngnod);

  for (int i = 0; i < nedges; ++i) {
    const int ispec = self_edges(i).ispec;
    const auto iedge_type = self_edges(i).edge_type;
    const int jspec = coupled_edges(i).ispec;
    const auto jedge_type = coupled_edges(i).edge_type;
    for (int i = 0; i < mesh.ngnod; i++) {
      icoorg(i).x = mesh.h_control_node_coord(0, ispec, i);
      icoorg(i).z = mesh.h_control_node_coord(1, ispec, i);
      jcoorg(i).x = mesh.h_control_node_coord(0, jspec, i);
      jcoorg(i).z = mesh.h_control_node_coord(1, jspec, i);
    }
    auto transfer_subview =
        Kokkos::subview(h_transfer_function, i, Kokkos::ALL, Kokkos::ALL);
    auto transfer_subview_other =
        Kokkos::subview(h_transfer_function_other, i, Kokkos::ALL, Kokkos::ALL);
    if (is_side1) {
      specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
          icoorg, jcoorg, iedge_type, jedge_type, interface_quadrature,
          mesh.h_xi, transfer_subview, transfer_subview_other);
    } else {
      specfem::assembly::nonconforming_interfaces_impl::set_transfer_functions(
          jcoorg, icoorg, jedge_type, iedge_type, interface_quadrature,
          mesh.h_xi, transfer_subview_other, transfer_subview);
    }

    const int npoints =
        connection_mapping.number_of_points_on_orientation(iedge_type);
    for (int ipoint = 0; ipoint < npoints; ++ipoint) {
      const auto [iz, ix] =
          connection_mapping.coordinates_at_edge(iedge_type, ipoint);
      specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                      false>
          point_jacobian_matrix;
      specfem::point::index<specfem::dimension::type::dim2, false> point_index{
        ispec, iz, ix
      };
      specfem::assembly::load_on_host(point_index, jacobian_matrix,
                                      point_jacobian_matrix);
      const auto dn = point_jacobian_matrix.compute_normal(iedge_type);
      const type_real mag = std::sqrt(dn(0) * dn(0) + dn(1) * dn(1));

      if (store_edge_normal) {
        this->h_edge_normal(i, ipoint, 0) = dn(0) / mag;
        this->h_edge_normal(i, ipoint, 1) = dn(1) / mag;
      }
      const std::array<type_real, 2> w{ weights(ix), weights(iz) };
      this->h_edge_factor(i, ipoint) = [&]() {
        switch (iedge_type) {
        case specfem::mesh_entity::type::bottom:
        case specfem::mesh_entity::type::top:
          return w[0] * mag;
        case specfem::mesh_entity::type::left:
        case specfem::mesh_entity::type::right:
          return w[1] * mag;
        default:
          KOKKOS_ABORT_WITH_LOCATION("Invalid edge type");
          return static_cast<type_real>(0.0);
        }
        return static_cast<type_real>(0.0);
      }();
    }
  }

  Kokkos::deep_copy(edge_factor, h_edge_factor);
  if (store_edge_normal) {
    Kokkos::deep_copy(edge_normal, h_edge_normal);
  }

  Kokkos::deep_copy(transfer_function, h_transfer_function);
  Kokkos::deep_copy(transfer_function_other, h_transfer_function_other);
}
