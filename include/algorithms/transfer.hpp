#pragma once

#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "execution/team_thread_md_range_iterator.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::algorithms {

/**
 * @brief Takes a chunk_edge::field and maps it onto the intersection, using a
 single-sided
 * transfer-function container.
 *
 * @tparam CoupledInterfaceType transfer function container type
 (specfem::assembly::coupled_interfaces_impl::stores_transfer_function_coupled<CoupledInterfaceType>::value
 must be true)
 * @tparam EdgeFieldType The chunk_edge field type
 * @tparam IntersectionFieldViewType - a view that the intersection field should
 be stored into
 * @param interface_data transfer function container
 * @param coupled_field The chunk_edge field to map from
 * @param intersection_field a view that the intersection field should be stored
 into
 */
template <
    typename IndexType, typename TransferFunctionType, typename EdgeFieldType,
    typename IntersectionReturnCallback,
    typename std::enable_if_t<TransferFunctionType::connection_tag ==
                                  specfem::connections::type::nonconforming,
                              int> = 0>
KOKKOS_INLINE_FUNCTION void
transfer(const IndexType &chunk_edge_index,
         const TransferFunctionType &transfer_function,
         const EdgeFieldType &coupled_field,
         const IntersectionReturnCallback &callback) {

  constexpr auto dimension_tag = EdgeFieldType::dimension_tag;
  constexpr auto edge_medium_tag = EdgeFieldType::medium_tag;
  constexpr auto interface_tag = TransferFunctionType::interface_tag;

  static_assert(
      specfem::data_access::is_chunk_edge<IndexType>::value,
      "The index for a nonconforming compute_coupling must be a chunk_edge.");

  static_assert(specfem::data_access::is_chunk_edge<EdgeFieldType>::value &&
                    specfem::data_access::is_field<EdgeFieldType>::value,
                "coupled_field is not a point field type");

  // TODO future consideration: use load_on_device for coupled field here.
  // We would want it to be a specialization, since we want to transfer more
  // things than just fields is there a better way of recovering global index?
  const auto &team = chunk_edge_index.get_policy_index();
  const int &num_edges = chunk_edge_index.nedges();

  using VectorPointViewType = specfem::datatype::VectorPointViewType<
      type_real, EdgeFieldType::components, EdgeFieldType::using_simd>;

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, edge_medium_tag>::components;

  specfem::execution::for_each_level(
      specfem::execution::TeamThreadMDRangeIterator(
          team, num_edges, TransferFunctionType::n_quad_intersection),
      [&](const auto &index) {
        const int iedge = index(0);
        const int iquad = index(1);
        VectorPointViewType intersection_point_view;

        for (int icomp = 0; icomp < ncomp; icomp++) {
          intersection_point_view(icomp) = 0;

          for (int ipoint_edge = 0;
               ipoint_edge < TransferFunctionType::n_quad_element;
               ipoint_edge++) {
            intersection_point_view(icomp) +=
                coupled_field(iedge, ipoint_edge, icomp) *
                transfer_function(iedge, ipoint_edge, iquad);
          }
        }

        callback(index, intersection_point_view);
      });
}

} // namespace specfem::algorithms
