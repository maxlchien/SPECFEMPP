#include "periodic_tasks/wavefield_writer.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"

// Explicit instantiation
template class specfem::periodic_tasks::wavefield_writer<specfem::io::HDF5>;

template class specfem::periodic_tasks::wavefield_writer<specfem::io::ASCII>;
