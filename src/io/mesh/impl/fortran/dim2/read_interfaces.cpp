#include "io/mesh/impl/fortran/dim2/read_interfaces.hpp"
#include "io/fortranio/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem_mpi/interface.hpp"

std::vector<std::array<int, 2> >
specfem::io::mesh::impl::fortran::dim2::read_interfaces(
    const int num_interfaces,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    std::ifstream &stream, const specfem::MPI::MPI *mpi) {

  int medium1_ispec_l, medium2_ispec_l;
  std::vector<std::array<int, 2> > ispecs(num_interfaces);

  for (int i = 0; i < num_interfaces; i++) {
    specfem::io::fortran_read_line(stream, &medium2_ispec_l, &medium1_ispec_l);
    ispecs[i] = { medium1_ispec_l - 1, medium2_ispec_l - 1 };
  }

  return ispecs;
}

std::vector<std::vector<std::array<int, 2> > >
specfem::io::mesh::impl::fortran::dim2::read_coupled_interfaces(
    std::ifstream &stream, const int num_interfaces_elastic_acoustic,
    const int num_interfaces_acoustic_poroelastic,
    const int num_interfaces_elastic_poroelastic,
    Kokkos::View<int **, Kokkos::LayoutRight, Kokkos::HostSpace> knods,
    const specfem::MPI::MPI *mpi) {

  std::vector<std::vector<std::array<int, 2> > > coupled_interfaces = {
    specfem::io::mesh::impl::fortran::dim2::read_interfaces(
        num_interfaces_elastic_acoustic, knods, stream, mpi),

    specfem::io::mesh::impl::fortran::dim2::read_interfaces(
        num_interfaces_acoustic_poroelastic, knods, stream, mpi),

    specfem::io::mesh::impl::fortran::dim2::read_interfaces(
        num_interfaces_elastic_poroelastic, knods, stream, mpi)
  };

  return coupled_interfaces;
}
