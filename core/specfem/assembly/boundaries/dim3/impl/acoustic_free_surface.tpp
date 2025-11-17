#pragma once

#include <map>
#include <stdexcept>
#include <vector>

#include "acoustic_free_surface.hpp"
#include "specfem/macros.hpp"
#include "utilities.hpp"

specfem::assembly::boundaries_impl::acoustic_free_surface<specfem::dimension::type::dim3>::
    acoustic_free_surface(
        const int nspec, const int ngllz, const int nglly, const int ngllx,
        const specfem::mesh::acoustic_free_surface<dimension_tag>
            &acoustic_free_surface,
        const specfem::assembly::mesh<dimension_tag> &mesh,
        const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
        std::vector<specfem::element::boundary_tag_container>
            &element_boundary_tags,
        const specfem::mesh::tags<dimension_tag> &mesh_tags) {

  // For 3D, the mesh structure already contains all boundary information:
  // - ispec(i): spectral element index for each face
  // - ijk(i, dim, gll_point): GLL point indices for each face
  // - normal(i, dim, gll_point): pre-computed normal vectors
  // - jacobian2Dw(i, gll_point): pre-computed 2D jacobian with weight

  // We just need to create a mapping from ispec to boundary index
  // and mark which quadrature points are on the boundary

  // -------------------------------------------------------------------

  // Create a map from ispec to index in acoustic_free_surface

  const int nelements = acoustic_free_surface.nelements;

  // Initialize all index mappings to -1
  for (int ispec = 0; ispec < nspec; ++ispec) {
    boundary_index_mapping(ispec) = -1;
  }

  // Early return if no free surface faces
  if (nelements == 0) {
    return;
  }

  std::map<int, std::vector<int> > ispec_to_acoustic_surface;

  for (int i = 0; i < nelements; ++i) {
    const int ispec = acoustic_free_surface.ispec(i);
    if (ispec_to_acoustic_surface.find(ispec) ==
        ispec_to_acoustic_surface.end()) {
      ispec_to_acoustic_surface[ispec] = { i };
    } else {
      ispec_to_acoustic_surface[ispec].push_back(i);
    }
  }

  const int total_acfree_surface_elements = ispec_to_acoustic_surface.size();

  // -------------------------------------------------------------------

  // Assign boundary index mapping
  int total_indices = 0;
  for (auto &map : ispec_to_acoustic_surface) {
    const int ispec = map.first;
    boundary_index_mapping(ispec) = total_indices;
    ++total_indices;
  }

  ASSERT(total_indices == total_acfree_surface_elements,
         "Total indices do not match");

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
      throw std::runtime_error("Boundary index mapping is not contiguous");
    }
  }

  // -------------------------------------------------------------------

  // Initialize boundary tags
  this->quadrature_point_boundary_tag =
      BoundaryTagView("specfem::assembly::impl::boundaries::"
                      "acoustic_free_surface::quadrature_point_boundary_tag",
                      total_indices, ngllz, nglly, ngllx);

  this->h_quadrature_point_boundary_tag =
      Kokkos::create_mirror_view(quadrature_point_boundary_tag);

  // Assign boundary tags
  // For 3D, we use the ijk array to determine which GLL points are on the boundary

  for (auto &map : ispec_to_acoustic_surface) {
    const int ispec = map.first;
    const auto &indices = map.second;
    const int index_compute = boundary_index_mapping(ispec);

    // Only add the acoustic_free_surface tag if the element is actually acoustic
    // Elastic elements at the free surface do not require special treatment
    const auto medium_tag = mesh_tags.tags_container(ispec).medium_tag;
    if (medium_tag == specfem::element::medium_tag::acoustic) {
      element_boundary_tags[ispec] +=
          specfem::element::boundary_tag::acoustic_free_surface;
    }

    // For each face on this element
    for (auto &index : indices) {
      // The ijk array contains the i,j,k indices for each GLL point on the face
      // ijk(index, 0:2, igll) gives the i,j,k indices for GLL point igll on face index
      for (int igll = 0; igll < acoustic_free_surface.ngllsquare; ++igll) {
        const int ix = acoustic_free_surface.ijk(index, 0, igll);
        const int iy = acoustic_free_surface.ijk(index, 1, igll);
        const int iz = acoustic_free_surface.ijk(index, 2, igll);

        this->h_quadrature_point_boundary_tag(index_compute, iz, iy, ix) +=
            specfem::element::boundary_tag::acoustic_free_surface;
      }
    }
  }

  Kokkos::deep_copy(quadrature_point_boundary_tag,
                    h_quadrature_point_boundary_tag);
}
