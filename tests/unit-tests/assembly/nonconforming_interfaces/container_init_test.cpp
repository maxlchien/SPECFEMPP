#include "../../MPI_environment.hpp"
#include "io/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/assembly/nonconforming_interfaces/dim2/impl/compute_intersection.tpp"
#include "specfem/point.hpp"
#include "specfem/point/coordinates.hpp"
#include <gtest/gtest.h>
#include <memory>

template <typename TransferView>
specfem::point::global_coordinates<specfem::dimension::type::dim2>
transfer_position_to_mortar(
    const TransferView &transfer_function,
    const Kokkos::View<type_real ****, Kokkos::LayoutRight,
                       Kokkos::DefaultExecutionSpace> &coord_view,
    const int &ispec, const specfem::mesh_entity::type &edge_type,
    const int &iedge, const int &imortar) {
  specfem::point::global_coordinates<specfem::dimension::type::dim2> coords(0,
                                                                            0);
  const int ngllz = coord_view.extent(2);
  const int ngllx = coord_view.extent(3);
  const auto connection_mapping =
      specfem::connections::connection_mapping(ngllx, ngllz);

  for (int ipoint = 0; ipoint < transfer_function.extent(2); ipoint++) {
    const auto [iz, ix] =
        connection_mapping.coordinates_at_edge(edge_type, ipoint);
    const type_real transfer = transfer_function(iedge, imortar, ipoint);
    coords.x += transfer * coord_view(0, ispec, iz, ix);
    coords.z += transfer * coord_view(1, ispec, iz, ix);
  }
  return coords;
}

void test_nonconforming_container_transfers(
    const specfem::assembly::assembly<specfem::dimension::type::dim2>
        &assembly) {
  const int ngllx = assembly.mesh.shape_functions::ngllx;
  const int ngllz = assembly.mesh.shape_functions::ngllz;

  const auto &nc_interface_acoustic_elastic =
      assembly.coupled_interfaces.get_nonconforming_interface_container<
          specfem::interface::interface_tag::acoustic_elastic,
          specfem::element::boundary_tag::none>();

  const auto &nc_interface_elastic_acoustic =
      assembly.coupled_interfaces.get_nonconforming_interface_container<
          specfem::interface::interface_tag::elastic_acoustic,
          specfem::element::boundary_tag::none>();

  const auto [acoustic_edges, elastic_edges] =
      assembly.edge_types.get_edges_on_host(
          specfem::connections::type::nonconforming,
          specfem::interface::interface_tag::acoustic_elastic,
          specfem::element::boundary_tag::none);

  const int nedges = acoustic_edges.size();
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(0), nedges)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct number of intersections.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(0),
            nedges)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct number of intersections.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(0),
            nedges)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct number of intersections.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(0), nedges)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct number of intersections.";

  const int ngll = ngllz;
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(2), ngll)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(2),
            ngll)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(2),
            ngll)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct number of edge quadrature points.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(2), ngll)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct number of edge quadrature points.";

  // properly set this once we get mortar quad parameter up
  const int nquad_mortar =
      nc_interface_acoustic_elastic.h_transfer_function.extent(1);
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function.extent(1),
            nquad_mortar)
      << "acoustic side of the the acoustic-elastic interface does not have "
         "the correct interface quadrature size.";
  ASSERT_EQ(nc_interface_acoustic_elastic.h_transfer_function_other.extent(1),
            nquad_mortar)
      << "elastic side of the the acoustic-elastic interface does not have the "
         "correct interface quadrature size.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function_other.extent(1),
            nquad_mortar)
      << "acoustic side of the the elastic-acoustic interface does not have "
         "the correct interface quadrature size.";
  ASSERT_EQ(nc_interface_elastic_acoustic.h_transfer_function.extent(1),
            nquad_mortar)
      << "elastic side of the the elastic-acoustic interface does not have the "
         "correct interface quadrature size.";

  for (int iedge = 0; iedge < nedges; iedge++) {
    for (int imortar = 0; imortar < nquad_mortar; imortar++) {
      const int ac_spec = acoustic_edges(iedge).ispec;
      const specfem::mesh_entity::type ac_edgetype =
          acoustic_edges(iedge).edge_type;
      const int el_spec = elastic_edges(iedge).ispec;
      const specfem::mesh_entity::type el_edgetype =
          elastic_edges(iedge).edge_type;

      // interpolated positions on acoustic-elastic interface
      const auto acel_self = transfer_position_to_mortar(
          nc_interface_acoustic_elastic.h_transfer_function,
          assembly.mesh.h_coord, ac_spec, ac_edgetype, iedge, imortar);
      const auto acel_other = transfer_position_to_mortar(
          nc_interface_acoustic_elastic.h_transfer_function_other,
          assembly.mesh.h_coord, el_spec, el_edgetype, iedge, imortar);
      // interpolated positions on elastic-acoustic interface
      const auto elac_self = transfer_position_to_mortar(
          nc_interface_elastic_acoustic.h_transfer_function,
          assembly.mesh.h_coord, el_spec, el_edgetype, iedge, imortar);
      const auto elac_other = transfer_position_to_mortar(
          nc_interface_elastic_acoustic.h_transfer_function_other,
          assembly.mesh.h_coord, ac_spec, ac_edgetype, iedge, imortar);
      EXPECT_TRUE(specfem::point::distance(acel_self, acel_other) < 1e-3)
          << "acoustic-elastic interface has transfer functions from either "
             "side mapping to incompatible coordinates.";
      EXPECT_TRUE(specfem::point::distance(elac_self, elac_other) < 1e-3)
          << "elastic-acoustic interface has transfer functions from either "
             "side mapping to incompatible coordinates.";
      EXPECT_TRUE(specfem::point::distance(acel_self, elac_self) < 1e-3)
          << "elastic-acoustic and acoustic-elastic interface do not agree in "
             "coordinates.";
    }
  }
}

TEST(NonconformingInterfaces, ContainerInitialization) {
  std::string database_file("data/mesh/3_elem_nonconforming/database.bin");
  specfem::MPI::MPI *mpi = MPIEnvironment::get_mpi();

  const auto mesh = specfem::io::read_2d_mesh(
      database_file, specfem::enums::elastic_wave::psv,
      specfem::enums::electromagnetic_wave::te, mpi);

  const auto quadrature = []() {
    specfem::quadrature::gll::gll gll{};
    return specfem::quadrature::quadratures(gll);
  }();

  std::vector<std::shared_ptr<
      specfem::sources::source<specfem::dimension::type::dim2> > >
      sources;
  std::vector<std::shared_ptr<
      specfem::receivers::receiver<specfem::dimension::type::dim2> > >
      receivers;
  specfem::assembly::assembly<specfem::dimension::type::dim2> assembly(
      mesh, quadrature, sources, receivers, {}, 1.0, 0.0, 1, 1, 1,
      specfem::simulation::type::forward, false, nullptr);

  // consider adding more assemblies to this test
  test_nonconforming_container_transfers(assembly);
}
