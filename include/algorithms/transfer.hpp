#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

// TODO replace when this gets reworked.
#include "specfem/assembly/coupled_interfaces/dim2/data_access/impl/load_access_compatibility.hpp"
namespace specfem::algorithms {

/**
 * @brief Takes a chunk_edge::field and maps it onto the intersection, using a
 single-sided
 * transfer-function container.
 *
 * @tparam CoupledInterfaceType transfer function container type
 (specfem::assembly::coupled_interfaces_impl::stores_transfer_function_single_side<CoupledInterfaceType>::value
 must be true)
 * @tparam EdgeFieldType The chunk_edge field type
 * @tparam IntersectionFieldViewType - a view that the intersection field should
 be stored into
 * @param interface_data transfer function container
 * @param coupled_field The chunk_edge field to map from
 * @param intersection_field a view that the intersection field should be stored
 into
 */
template <typename CoupledInterfaceType, typename EdgeFieldType,
          typename IntersectionFieldViewType,
          typename std::enable_if_t<
              CoupledInterfaceType::connection_tag ==
                      specfem::connections::type::nonconforming &&
                  specfem::assembly::coupled_interfaces_impl::
                      stores_transfer_function_single_side<
                          CoupledInterfaceType>::value,
              int> = 0>
KOKKOS_INLINE_FUNCTION void
transfer(const CoupledInterfaceType &interface_data,
         const EdgeFieldType &coupled_field,
         IntersectionFieldViewType &intersection_field) {

  constexpr auto dimension_tag = EdgeFieldType::dimension_tag;
  constexpr auto edge_medium_tag = EdgeFieldType::medium_tag;
  constexpr auto interface_tag = CoupledInterfaceType::interface_tag;

  static_assert(
      specfem::data_access::is_coupled_interface<CoupledInterfaceType>::value,
      "interface_data is not a coupled interface type");
  static_assert(specfem::data_access::is_chunk_edge<EdgeFieldType>::value &&
                    specfem::data_access::is_field<EdgeFieldType>::value,
                "coupled_field is not a point field type");

  // no medium check for intersection.
  static_assert(
      (specfem::assembly::coupled_interfaces_impl::
               stores_transfer_function_self<CoupledInterfaceType>::value
           ? specfem::interface::attributes<dimension_tag,
                                            interface_tag>::self_medium()
           : specfem::interface::attributes<dimension_tag,
                                            interface_tag>::coupled_medium()) ==
          edge_medium_tag,
      "Inconsistent medium tag between CoupledInterfaceType's side of the "
      "interface and EdgeFieldType");

  // check number of axes match
  // This will change.
  static_assert(IntersectionFieldViewType::rank == 3,
                "IntersectionFieldViewType must be a view with 3 axes: "
                "view(local_iedge, ipoint, field_idim)");

  constexpr int ncomp =
      specfem::element::attributes<dimension_tag, edge_medium_tag>::components;

  const auto &transfer_function = interface_data.get_transfer_function();

  // TODO decide how to handle TeamThreadRange passing for inner parfor here
  for (int iedge = 0; iedge < CoupledInterfaceType::chunk_size; iedge++) {
    for (int ipoint_intersection = 0;
         ipoint_intersection < CoupledInterfaceType::n_quad_intersection;
         ipoint_intersection++) {
      for (int icomp = 0; icomp < ncomp; icomp++) {
        intersection_field(iedge, ipoint_intersection, icomp) = 0;

        for (int ipoint_edge = 0;
             ipoint_edge < CoupledInterfaceType::n_quad_element;
             ipoint_edge++) {
          intersection_field(iedge, ipoint_intersection, icomp) +=
              coupled_field(iedge, ipoint_edge, icomp) *
              transfer_function(iedge, ipoint_edge, ipoint_intersection);
        }
      }
    }
  }
}

} // namespace specfem::algorithms
