#include "kokkos_kernels/impl/compute_material_derivatives.hpp"
#include "kokkos_kernels/impl/compute_material_derivatives.tpp"

FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
     MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC),
     PROPERTY_TAG(ISOTROPIC, ANISOTROPIC)),
    INSTANTIATE(
        /** instantiation for NGLL = 5     */
        (template void
             specfem::kokkos_kernels::impl::compute_material_derivatives,
         (_DIMENSION_TAG_, 5, _MEDIUM_TAG_, _PROPERTY_TAG_),
         (const specfem::compute::assembly &, const type_real &);),
        /** instantiation for NGLL = 8     */
        (template void
             specfem::kokkos_kernels::impl::compute_material_derivatives,
         (_DIMENSION_TAG_, 8, _MEDIUM_TAG_, _PROPERTY_TAG_),
         (const specfem::compute::assembly &, const type_real &);)))
