#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point.hpp"

namespace specfem::assembly::boundaries_impl {

template <> struct stacey<specfem::dimension::type::dim3> {
private:
  constexpr static auto boundary_tag =
      specfem::element::boundary_tag::stacey; ///< Boundary tag

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension
  using BoundaryTagView =
      Kokkos::View<specfem::element::boundary_tag_container ****,
                   Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace>;
  using FaceNormalView = Kokkos::View<type_real *****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;
  using FaceWeightView = Kokkos::View<type_real ****, Kokkos::LayoutLeft,
                                      Kokkos::DefaultExecutionSpace>;

  BoundaryTagView quadrature_point_boundary_tag; ///< Boundary tag for every
  ///< quadrature point within an
  ///< element with Stacey boundary

  BoundaryTagView::HostMirror h_quadrature_point_boundary_tag; ///< Host mirror
                                                               ///< of boundary
                                                               ///< types

  FaceNormalView face_normal; ///< Normal vector to the face for every
                              ///< quadrature point within an element with
                              ///< Stacey boundary
  FaceWeightView face_weight; ///< Face weight used to compute integrals on the
                              ///< face for every quadrature point within an
                              ///< element with Stacey boundary. Evaluates to 0
                              ///< for points not on the face

  FaceNormalView::HostMirror h_face_normal; ///< Host mirror of face normal

  FaceWeightView::HostMirror h_face_weight; ///< Host mirror of face weight

  stacey() = default;

  stacey(
      const int nspec, const int ngllz, const int nglly, const int ngllx,
      const specfem::mesh::absorbing_boundary<dimension_tag> &stacey,
      const specfem::assembly::mesh<dimension_tag> &mesh,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const Kokkos::View<int *, Kokkos::HostSpace> &boundary_index_mapping,
      std::vector<specfem::element::boundary_tag_container> &boundary_tag);

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<boundary_tag, dimension_tag, false>
                     &boundary) const {

    boundary.tag += quadrature_point_boundary_tag(index.ispec, index.iz,
                                                  index.iy, index.ix);

    boundary.face_normal(0) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        face_weight(index.ispec, index.iz, index.iy, index.ix);

    return;
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension_tag, false> &boundary) const {

    boundary.tag += quadrature_point_boundary_tag(index.ispec, index.iz,
                                                  index.iy, index.ix);

    boundary.face_normal(0) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        face_normal(index.ispec, index.iz, index.iy, index.ix, 2);
    boundary.face_weight =
        face_weight(index.ispec, index.iz, index.iy, index.ix);

    return;
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<boundary_tag, dimension_tag, true>
                     &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.face_normal(0))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(1))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(2))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 2),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_weight)
        .copy_from(&face_weight(index.ispec, index.iz, index.iy, index.ix),
                   tag_type());
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  KOKKOS_FORCEINLINE_FUNCTION void
  load_on_device(const IndexType &index,
                 specfem::point::boundary<
                     specfem::element::boundary_tag::composite_stacey_dirichlet,
                     dimension_tag, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.face_normal(0))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(1))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(2))
        .copy_from(&face_normal(index.ispec, index.iz, index.iy, index.ix, 2),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_weight)
        .copy_from(&face_weight(index.ispec, index.iz, index.iy, index.ix),
                   tag_type());
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    false> &boundary) const {
    boundary.tag += h_quadrature_point_boundary_tag(index.ispec, index.iz,
                                                    index.iy, index.ix);

    boundary.face_normal(0) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2);

    boundary.face_weight =
        h_face_weight(index.ispec, index.iz, index.iy, index.ix);

    return;
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == false,
                int> = 0>
  inline void
  load_on_host(const IndexType &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension_tag, false> &boundary) const {
    boundary.tag += h_quadrature_point_boundary_tag(index.ispec, index.iz,
                                                    index.iy, index.ix);

    boundary.face_normal(0) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0);
    boundary.face_normal(1) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1);
    boundary.face_normal(2) =
        h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2);

    boundary.face_weight =
        h_face_weight(index.ispec, index.iz, index.iy, index.ix);

    return;
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  inline void load_on_host(const IndexType &index,
                           specfem::point::boundary<boundary_tag, dimension_tag,
                                                    true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.face_normal(0))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(1))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(2))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_weight)
        .copy_from(&h_face_weight(index.ispec, index.iz, index.iy, index.ix),
                   tag_type());

    return;
  }

  template <typename IndexType,
            typename std::enable_if_t<
                specfem::data_access::is_index_type<IndexType>::value &&
                    specfem::data_access::is_point<IndexType>::value &&
                    IndexType::using_simd == true,
                int> = 0>
  inline void
  load_on_host(const IndexType &index,
               specfem::point::boundary<
                   specfem::element::boundary_tag::composite_stacey_dirichlet,
                   dimension_tag, true> &boundary) const {

    using simd = typename specfem::datatype::simd<type_real, true>;

    using mask_type = typename simd::mask_type;
    using tag_type = typename simd::tag_type;

    mask_type mask([&](std::size_t lane) { return index.mask(lane); });

    for (int lane = 0; lane < mask_type::size(); ++lane) {
      if (index.mask(lane)) {
        boundary.tag[lane] += h_quadrature_point_boundary_tag(
            index.ispec + lane, index.iz, index.iy, index.ix);
      }
    }

    Kokkos::Experimental::where(mask, boundary.face_normal(0))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 0),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(1))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 1),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_normal(2))
        .copy_from(&h_face_normal(index.ispec, index.iz, index.iy, index.ix, 2),
                   tag_type());

    Kokkos::Experimental::where(mask, boundary.face_weight)
        .copy_from(&h_face_weight(index.ispec, index.iz, index.iy, index.ix),
                   tag_type());

    return;
  }
};
} // namespace specfem::assembly::boundaries_impl
