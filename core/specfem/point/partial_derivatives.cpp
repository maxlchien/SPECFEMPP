
#include "partial_derivatives.hpp"
#include "partial_derivatives.tpp"
// Explicit template instantiation

template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, false, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, true, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, false, true>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim2, true, true>;

template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim3, false, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim3, true, false>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim3, false, true>;
template struct specfem::point::partial_derivatives<
    specfem::dimension::type::dim3, true, true>;
