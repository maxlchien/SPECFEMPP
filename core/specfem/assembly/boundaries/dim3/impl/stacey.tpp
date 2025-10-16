#pragma once

#include <array>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "enumerations/interface.hpp"
#include "macros.hpp"
#include "stacey.hpp"
#include "utilities.hpp"
#include <Kokkos_Core.hpp>

specfem::assembly::boundaries_impl::stacey<specfem::dimension::type::dim3>::
    stacey(const int nspec, const int ngllz, const int nglly, const int ngllx,
           const specfem::mesh::absorbing_boundary<dimension_tag> &stacey,
           const specfem::assembly::mesh<dimension_tag> &mesh,
           const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
           const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
           std::vector<specfem::element::boundary_tag_container>
               &element_boundary_tags) {

  // For 3D, the mesh structure already contains all boundary information:
  // - ispec(i): spectral element index for each face
  // - ijk(i, dim, gll_point): GLL point indices for each face
  // - normal(i, dim, gll_point): pre-computed normal vectors
  // - jacobian2Dw(i, gll_point): pre-computed 2D jacobian with weight

  // We just need to create a mapping from ispec to boundary index
  // and use the pre-computed boundary data

  // -------------------------------------------------------------------

  // Create a map from ispec to index in stacey

  const int nelements = stacey.num_abs_boundary_faces;

  // Initialize all index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Early return if no absorbing boundaries (ABC is disabled)
  // Even if the mesh file contains boundary data, we should skip processing
  // if num_abs_boundary_faces is 0 or if elastic/acoustic flags are both false
  if (nelements == 0 || (!stacey.elastic && !stacey.acoustic)) {
    return;
  }

  std::map<int, std::vector<int> > ispec_to_stacey;

  for (int i = 0; i < nelements; ++i) {
    const int ispec = stacey.ispec(i);
    if (ispec_to_stacey.find(ispec) == ispec_to_stacey.end()) {
      ispec_to_stacey[ispec] = { i };
    } else {
      ispec_to_stacey[ispec].push_back(i);
    }
  }

  const int total_stacey_elements = ispec_to_stacey.size();

  // -------------------------------------------------------------------

  // Assign boundary index mapping

  // Assign boundary index mapping
  int total_indices = 0;

  for (auto &map : ispec_to_stacey) {
    const int ispec = map.first;
    boundary_index_mapping(ispec) = total_indices;
    ++total_indices;
  }

  ASSERT(total_indices == total_stacey_elements,
         "Error: Total number of Stacey elements do not match");

  // -------------------------------------------------------------------
  // Make sure the index mapping is contiguous

  for (int ispec = 0; ispec < nspec; ++ispec) {
    if (ispec == 0)
      continue;

    if ((boundary_index_mapping(ispec) == -1) ||
        (boundary_index_mapping(ispec - 1) == -1))
      continue;

    if (boundary_index_mapping(ispec) !=
        boundary_index_mapping(ispec - 1) + 1) {

      std::cout << "ispec: " << ispec << std::endl;
      std::cout << "boundary_index_mapping(ispec): "
                << boundary_index_mapping(ispec) << std::endl;
      std::cout << "boundary_index_mapping(ispec - 1): "
                << boundary_index_mapping(ispec - 1) << std::endl;

      throw std::runtime_error("Boundary index mapping is not contiguous");
    }
  }

  // -------------------------------------------------------------------

  // -------------------------------------------------------------------

  // Initialize views

  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::stacey::"
                      "quadrature_point_boundary_tag",
                      total_stacey_elements, ngllz, nglly, ngllx);

  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  this->face_weight = FaceWeightView("specfem::assembly::impl::boundaries::"
                                     "stacey::face_weight",
                                     total_stacey_elements, ngllz, nglly, ngllx);

  this->face_normal = FaceNormalView("specfem::assembly::impl::boundaries::"
                                     "stacey::face_normal",
                                     total_stacey_elements, ngllz, nglly, ngllx, 3);

  this->h_face_weight = Kokkos::create_mirror_view(face_weight);
  this->h_face_normal = Kokkos::create_mirror_view(face_normal);

  // -------------------------------------------------------------------

  // Assign boundary values
  // For 3D, we use the pre-computed normal and jacobian2Dw from the mesh

  for (auto &map : ispec_to_stacey) {
    const int ispec = map.first;
    const auto &indices = map.second;
    const int local_index = boundary_index_mapping(ispec);

    element_boundary_tags[ispec] +=
        specfem::element::boundary_tag::stacey;

    // For each face on this element
    for (int i : indices) {
      // The ijk array contains the i,j,k indices for each GLL point on the face
      // ijk(i, 0:2, igll) gives the i,j,k indices for GLL point igll on face i
      for (int igll = 0; igll < stacey.ngllsquare; ++igll) {
        const int ix = stacey.ijk(i, 0, igll);
        const int iy = stacey.ijk(i, 1, igll);
        const int iz = stacey.ijk(i, 2, igll);

        this->h_quadrature_point_boundary_tag(local_index, iz, iy, ix) +=
            specfem::element::boundary_tag::stacey;

        // Use pre-computed normal and jacobian from mesh
        this->h_face_weight(local_index, iz, iy, ix) = stacey.jacobian2Dw(i, igll);
        this->h_face_normal(local_index, iz, iy, ix, 0) = stacey.normal(i, 0, igll);
        this->h_face_normal(local_index, iz, iy, ix, 1) = stacey.normal(i, 1, igll);
        this->h_face_normal(local_index, iz, iy, ix, 2) = stacey.normal(i, 2, igll);
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
  Kokkos::deep_copy(face_weight, h_face_weight);
  Kokkos::deep_copy(face_normal, h_face_normal);
}
