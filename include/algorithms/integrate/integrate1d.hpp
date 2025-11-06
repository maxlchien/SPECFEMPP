#include <Kokkos_Core.hpp>

namespace specfem::algorithms {

/**
 * @brief Takes a field `intersection_field` on the intersection and computes,
 * for each self GLL point, the integral of `intersection_field` times the shape
 * function at that point. `intersection_field` should be call-accessible (e.g.
 * Kokkos::View) with shape:
 *
 * (chunk_size, n_quad_intersection, self::components())
 *
 * After handling any other intersection forces, boundary conditions, etc. the
 * result can be `atomic_add`ed to the acceleration field.
 *
 * @tparam dimension_tag dimension of the simulation
 * @tparam IndexType The chunk_edge iterator type
 * @tparam IntersectionFieldViewType The type of `intersection_field`
 * @tparam ChunkEdgeWeightJacobianType A nonconforming chunk_edge accessor
 * holding `intersection_factor`
 * @tparam CallableType The callback function, which will be given the point
 * index and corresponding evaluated integral
 * @param assembly - assembly struct
 * @param chunk_index - the outer index (chunk_edge) that gets iterated for
 * points
 * @param intersection_field - the field to integrate
 * @param weight_jacobian - nonconforming chunk_edge accessor holding
 * `intersection_factor`
 * @param callback - callback function to capture integral values
 */
template <specfem::dimension::type dimension_tag, typename IndexType,
          typename IntersectionFieldViewType,
          typename ChunkEdgeWeightJacobianType, typename CallableType>
KOKKOS_FUNCTION void integrate_fieldtilde_1d(
    const specfem::assembly::assembly<dimension_tag> &assembly,
    const IndexType &chunk_index,
    const IntersectionFieldViewType &intersection_field,
    const ChunkEdgeWeightJacobianType &weight_jacobian,
    const CallableType &callback) {

  constexpr auto self_medium_tag = specfem::interface::attributes<
      dimension_tag, ChunkEdgeWeightJacobianType::interface_tag>::self_medium();

  using PointIndexType =
      typename IndexType::iterator_type::index_type::index_type;
  using PointFieldType =
      specfem::point::acceleration<dimension_tag, self_medium_tag,
                                   IntersectionFieldViewType::using_simd>;
  using SelfTransferFunctionType =
      typename specfem::point::nonconforming_transfer_function<
          true, ChunkEdgeWeightJacobianType::n_quad_intersection, dimension_tag,
          ChunkEdgeWeightJacobianType::connection_tag,
          ChunkEdgeWeightJacobianType::interface_tag,
          ChunkEdgeWeightJacobianType::boundary_tag>;

  // an is_invocable static check here prevents autos

  constexpr int ncomp = PointFieldType::components;
  constexpr int nquad_intersection =
      ChunkEdgeWeightJacobianType::n_quad_intersection;

  specfem::execution::for_each_level(
      chunk_index.get_iterator(),
      [&](const typename IndexType::iterator_type::index_type &index) {
        const auto self_index = index.get_index();

        auto self_index_with_global = self_index;
        self_index_with_global.iedge +=
            chunk_index.get_policy_index().league_rank() *
            ChunkEdgeWeightJacobianType::chunk_size;

        SelfTransferFunctionType transfer_function_self;
        specfem::assembly::load_on_device(self_index_with_global,
                                          assembly.coupled_interfaces,
                                          transfer_function_self);

        PointFieldType result;
#pragma unroll
        for (int icomp = 0; icomp < ncomp; icomp++) {
          result(icomp) = 0;
        }
        const int &iedge = self_index.iedge;
        const int &ipoint = self_index.ipoint;

#pragma unroll
        for (int iquad = 0; iquad < nquad_intersection; iquad++) {

#pragma unroll
          for (int icomp = 0; icomp < ncomp; icomp++) {
            result(icomp) +=
                intersection_field(iedge, iquad, icomp) *
                weight_jacobian.intersection_factor(iedge, iquad) *
                transfer_function_self.transfer_function_self(iquad);
          }
        }

        callback(self_index, result);
      });
}

} // namespace specfem::algorithms
