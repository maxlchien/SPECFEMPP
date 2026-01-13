#pragma once

#include "../../SPECFEM_Environment.hpp"
#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Pair.hpp"
#include "decl/Kokkos_Declare_SERIAL.hpp"
#include "enumerations/coupled_interface.hpp"
#include "enumerations/medium.hpp"
#include "medium/compute_coupling.hpp"
#include "medium/dim2/coupling_terms/acoustic_elastic.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/data_access/accessor.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <type_traits>

// We need to simulate a chunk_edge iteration:
template <specfem::dimension::type DimensionTag> class ChunkEdgeIndexSimulator {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndexSimulator(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;
  KokkosIndexType kokkos_index; ///< Kokkos team member for this chunk
};

template <specfem::dimension::type dimension_tag, typename EdgeFunction2D,
          typename Accessor>
struct EdgeFunctionWithEmbeddedAccessor
    : specfem::datatype::VectorChunkEdgeViewType<
          type_real, dimension_tag, 1, EdgeFunction2D::nquad_edge,
          EdgeFunction2D::num_components, false,
          typename EdgeFunction2D::memory_space, Kokkos::MemoryTraits<> >,
      Accessor {
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  static constexpr bool using_simd = false;

  template <typename Init>
  EdgeFunctionWithEmbeddedAccessor(const Init &init)
      : specfem::datatype::VectorChunkEdgeViewType<
            type_real, dimension_tag, 1, EdgeFunction2D::nquad_edge,
            EdgeFunction2D::num_components, false,
            typename EdgeFunction2D::memory_space, Kokkos::MemoryTraits<> >(
            init){};
};

template <specfem::interface::interface_tag interface_tag,
          typename EmbeddedEdgeFunctionAccessor, typename TransferFunction2D,
          typename IntersectionNormal2D, typename EdgeFunction2D,
          typename IntersectionFunction2D>
void execute_impl_compute_coupling(
    const TransferFunction2D &transfer_function,
    const IntersectionNormal2D &intersection_normal,
    const EdgeFunction2D &edge_function,
    const IntersectionFunction2D &expected_solution) {

  constexpr int num_edges = EdgeFunction2D::num_edges;
  constexpr auto dimension_tag = specfem::dimension::type::dim2;
  constexpr auto boundary_tag = specfem::element::boundary_tag::none;

  constexpr specfem::element::medium_tag self_medium =
      specfem::interface::attributes<dimension_tag,
                                     interface_tag>::self_medium();
  constexpr int ncomp_self =
      specfem::element::attributes<dimension_tag, self_medium>::components;

  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, TransferFunction2D::nquad_intersection,
      TransferFunction2D::nquad_edge,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, typename TransferFunction2D::memory_space,
      Kokkos::MemoryTraits<> >;
  using IntersectionNormalType = specfem::chunk_edge::intersection_normal<
      dimension_tag, interface_tag, boundary_tag, 1,
      TransferFunction2D::nquad_intersection,
      typename TransferFunction2D::memory_space, Kokkos::MemoryTraits<> >;

  using EdgeFunctionType =
      EdgeFunctionWithEmbeddedAccessor<dimension_tag, EdgeFunction2D,
                                       EmbeddedEdgeFunctionAccessor>;
  using ComputedCouplingFunction = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, TransferFunction2D::nquad_intersection,
      ncomp_self, false, typename TransferFunction2D::memory_space,
      Kokkos::MemoryTraits<> >;

  const auto transfer_function_view = transfer_function.get_view();
  const auto edge_function_view = edge_function.get_view();
  const auto intersection_normal_view = intersection_normal.get_view();

  const Kokkos::View<
      type_real * [TransferFunction2D::nquad_intersection][ncomp_self],
      typename TransferFunction2D::memory_space, Kokkos::MemoryTraits<> >
      computed_coupling_function("computed_coupling_function", num_edges);

  Kokkos::parallel_for(
      "impl_compute_coupling_test", Kokkos::TeamPolicy<>(num_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge_start = team_member.league_rank();
        const int iedge_end = team_member.league_rank() + 1;
        const int virtual_chunk_size = iedge_end - iedge_start;
        const auto view_slice = Kokkos::make_pair(iedge_start, iedge_end);
        const TransferFunctionType TF(Kokkos::subview(
            transfer_function_view, view_slice, Kokkos::ALL(), Kokkos::ALL()));
        const EdgeFunctionType EF(Kokkos::subview(
            edge_function_view, view_slice, Kokkos::ALL(), Kokkos::ALL()));
        const IntersectionNormalType IN(
            Kokkos::subview(intersection_normal_view, view_slice, Kokkos::ALL(),
                            Kokkos::ALL()));
        ComputedCouplingFunction CCF(Kokkos::subview(computed_coupling_function,
                                                     view_slice, Kokkos::ALL(),
                                                     Kokkos::ALL()));
        specfem::medium::impl::compute_coupling(
            std::integral_constant<specfem::dimension::type, dimension_tag>(),
            std::integral_constant<specfem::connections::type,
                                   specfem::connections::type::nonconforming>(),
            std::integral_constant<specfem::interface::interface_tag,
                                   interface_tag>(),
            ChunkEdgeIndexSimulator<dimension_tag>(num_edges, team_member), TF,
            IN, EF, CCF);
        for (int ielem = 0; ielem < virtual_chunk_size; ++ielem) {
          for (int ipoint = 0; ipoint < TransferFunction2D::nquad_intersection;
               ++ipoint) {
            for (int icomp = 0; icomp < ncomp_self; ++icomp) {
              computed_coupling_function(iedge_start + ielem, ipoint, icomp) =
                  CCF(ielem, ipoint, icomp);
            }
          }
        }
      });

  Kokkos::fence();
  typename decltype(
      computed_coupling_function)::HostMirror h_computed_coupling_function =
      Kokkos::create_mirror_view(computed_coupling_function);

  Kokkos::deep_copy(h_computed_coupling_function, computed_coupling_function);

  for (int ielem = 0; ielem < num_edges; ++ielem) {
    for (int ipoint = 0; ipoint < TransferFunction2D::nquad_intersection;
         ++ipoint) {
      for (int icomp = 0; icomp < ncomp_self; ++icomp) {
        const type_real got = computed_coupling_function(ielem, ipoint, icomp);
        const type_real expected = expected_solution(ielem, ipoint, icomp);

        if (!specfem::utilities::is_close(got, expected)) {
          std::ostringstream oss;
          oss << "-- Transfer function --\n"
              << TransferFunction2D::description() << std::endl
              << "-- Intersection Normal --\n"
              << IntersectionNormal2D::description() << std::endl
              << "-- Edge Function --\n"
              << EdgeFunction2D::description() << std::endl
              << "-- Expected Function --\n"
              << IntersectionFunction2D::description() << std::endl
              << "\n-- Failure --\n"
              << "Transfer function test failed at edge " << ielem
              << ": expected " << expected << "\n got " << got << std::endl;

          ADD_FAILURE() << oss.str();
        }
      }
    }
  }
}

template <specfem::interface::interface_tag InterfaceTag, typename = void>
struct EdgeFunctionAccessor {};

template <specfem::interface::interface_tag InterfaceTag>
struct EdgeFunctionAccessor<
    InterfaceTag,
    std::enable_if_t<InterfaceTag ==
                         specfem::interface::interface_tag::elastic_acoustic,
                     void> >
    : specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::acceleration,
          specfem::dimension::type::dim2, false> {};

template <specfem::interface::interface_tag InterfaceTag>
struct EdgeFunctionAccessor<
    InterfaceTag,
    std::enable_if_t<InterfaceTag ==
                         specfem::interface::interface_tag::acoustic_elastic,
                     void> >
    : specfem::data_access::Accessor<
          specfem::data_access::AccessorType::chunk_edge,
          specfem::data_access::DataClassType::displacement,
          specfem::dimension::type::dim2, false> {};
