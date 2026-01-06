
#include <array>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "algorithms/transfer.hpp"
#include "datatypes/chunk_edge_view.hpp"
#include "enumerations/interface.hpp"
#include "specfem/chunk_edge.hpp"
#include "specfem/data_access.hpp"
#include "utilities/include/fixture/nonconforming_interface.hpp"
#include "utilities/interface.hpp"

#include "SPECFEM_Environment.hpp"

namespace specfem::algorithms_test {

/**
 * @brief Test index type for chunk edge operations.
 */
class ChunkEdgeIndex {
public:
  static constexpr auto accessor_type =
      specfem::data_access::AccessorType::chunk_edge;
  using KokkosIndexType = Kokkos::TeamPolicy<>::member_type;

  /**
   * @brief Get Kokkos team member index.
   * @return Reference to Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  constexpr const KokkosIndexType &get_policy_index() const {
    return this->kokkos_index;
  }

  /**
   * @brief Construct chunk edge index.
   * @param nedges Number of edges in chunk
   * @param kokkos_index Kokkos team member
   */
  KOKKOS_INLINE_FUNCTION
  ChunkEdgeIndex(const int nedges, const KokkosIndexType &kokkos_index)
      : kokkos_index(kokkos_index), _nedges(nedges) {}

  /**
   * @brief Get number of edges.
   * @return Edge count
   */
  KOKKOS_INLINE_FUNCTION int nedges() const { return _nedges; }

private:
  int _nedges;                  ///< Number of edges in the chunk
  KokkosIndexType kokkos_index; /**< Kokkos team member for this chunk */
};

/** Test dimension (2D) */
constexpr static auto dimension_tag = specfem::dimension::type::dim2;
/** Interface type (dummy for testing) */
constexpr static auto interface_tag =
    specfem::interface::interface_tag::acoustic_elastic;
/** Boundary type (dummy for testing) */
constexpr static auto boundary_tag = specfem::element::boundary_tag::none;

template <typename T, typename = void>
struct is_analytical2d : std::false_type {};

template <typename T>
struct is_analytical2d<
    T, std::enable_if_t<T::FunctionInitializer::is_from_analytical_function,
                        void> > : std::true_type {};

/**
 * @brief Compute expected result of transfer function operation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param field Input field data
 * @return Expected transferred field values
 */
// TODO(Rohit : CPP20 update) Update this SFINAE with concepts
template <typename TransferFunction2D, typename EdgeFunction2D>
std::enable_if_t<
    !is_analytical2d<EdgeFunction2D>::value,
    std::vector<std::array<std::array<type_real, 1>,
                           TransferFunction2D::nquad_intersection> > >
expected_solution(const TransferFunction2D &transfer_function,
                  const EdgeFunction2D &field) {
  const int n_edges = TransferFunction2D::num_edges;
  std::vector<std::array<std::array<type_real, EdgeFunction2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
      result_field(
          n_edges,
          std::array<std::array<type_real, EdgeFunction2D::num_components>,
                     TransferFunction2D::nquad_intersection>{ 0.0 });
  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      for (int k = 0; k < EdgeFunction2D::num_components; ++k) {

        for (int l = 0; l < TransferFunction2D::nquad_edge; ++l) {
          result_field[i][j][k] += transfer_function(i, l, j) * field(i, l, k);
        }
      }
    }
  }
  throw std::runtime_error("No configs in TransferFunctionTestTypes2D uses "
                           "this. Remove if that changes.");
  return result_field;
}

/*
 * Specialization: We are transfering a function f =
 * AnalyticalFunction::evaluate using an actual quadrature rule.
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
std::enable_if_t<
    is_analytical2d<EdgeFunction2D>::value,
    std::vector<std::array<std::array<type_real, 1>,
                           TransferFunction2D::nquad_intersection> > >
expected_solution(const TransferFunction2D &transfer_function,
                  const EdgeFunction2D &field) {

  const int n_edges = TransferFunction2D::num_edges;
  std::vector<std::array<std::array<type_real, EdgeFunction2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
      result_field(
          n_edges,
          std::array<std::array<type_real, EdgeFunction2D::num_components>,
                     TransferFunction2D::nquad_intersection>{ 0.0 });
  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      for (int k = 0; k < EdgeFunction2D::num_components; ++k) {

        result_field[i][j][k] =
            EdgeFunction2D::FunctionInitializer::AnalyticalFunctionType::
                evaluate(TransferFunction2D::TransferFunctionInitializer::
                             intersection_quadrature_points[j])[k];
      }
    }
  }
  return result_field;
}

using ZeroTransferFunction = specfem::test_fixture::TransferFunction2D<
    specfem::test_fixture::TransferFunctionInitializer2D::Zero>;
using UniformEdgeFunction = specfem::test_fixture::EdgeFunction2D<
    specfem::test_fixture::EdgeFunctionInitializer2D::Uniform>;

/**
 * @brief Compute transferred field for zero transfer function and uniform
 * field.
 * @param transfer_function Zero transfer function
 * @param field Uniform field
 * @return Zero field result
 */

std::vector<
    std::array<std::array<type_real, UniformEdgeFunction::num_components>,
               ZeroTransferFunction::nquad_intersection> >
expected_solution(const ZeroTransferFunction &transfer_function,
                  const UniformEdgeFunction &field) {
  using TransferFunction2D = ZeroTransferFunction;
  using EdgeFunction2D = UniformEdgeFunction;
  // Result field is a zero field
  const int n_edges = EdgeFunction2D::num_edges;
  return std::vector<
      std::array<std::array<type_real, EdgeFunction2D::num_components>,
                 TransferFunction2D::nquad_intersection> >(n_edges, [] {
    std::array<std::array<type_real, EdgeFunction2D::num_components>,
               TransferFunction2D::nquad_intersection>
        arr{};
    return arr;
  }());
}

/**
 * @brief Execute transfer function test with validation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeFunction2D Field type
 * @param transfer_function Transfer function data
 * @param function Input function data
 */
template <typename TransferFunction2D, typename EdgeFunction2D>
void execute(const TransferFunction2D &transfer_function,
             const EdgeFunction2D &function) {
  auto expected = expected_solution(transfer_function, function);

  constexpr int n_edges = TransferFunction2D::num_edges;
  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, TransferFunction2D::nquad_intersection,
      TransferFunction2D::nquad_edge,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, typename TransferFunction2D::memory_space,
      Kokkos::MemoryTraits<> >;
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, TransferFunction2D::nquad_edge,
      EdgeFunction2D::num_components, false,
      typename TransferFunction2D::memory_space, Kokkos::MemoryTraits<> >;

  const auto transfer_function_view = transfer_function.get_view();
  const auto function_view = function.get_view();

  using ResultViewType =
      Kokkos::View<type_real[n_edges][TransferFunction2D::nquad_intersection]
                            [EdgeFunction2D::num_components],
                   typename TransferFunction2D::memory_space,
                   Kokkos::MemoryTraits<> >;

  ResultViewType result_view("result_view");

  Kokkos::parallel_for(
      "transfer_function_test", Kokkos::TeamPolicy<>(n_edges, 1, 1),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member) {
        const int iedge = team_member.league_rank();
        const TransferFunctionType TF(Kokkos::subview(
            transfer_function_view, Kokkos::make_pair(iedge, iedge + 1),
            Kokkos::ALL(), Kokkos::ALL()));
        const FunctionType F(
            Kokkos::subview(function_view, Kokkos::make_pair(iedge, iedge + 1),
                            Kokkos::ALL(), Kokkos::ALL()));
        specfem::algorithms::transfer(
            ChunkEdgeIndex(1, team_member), TF, F,
            [&](const auto &index, const auto &point) {
              for (int icomp = 0; icomp < EdgeFunction2D::num_components;
                   ++icomp) {
                Kokkos::single(Kokkos::PerTeam(team_member), [&]() {
                  result_view(index(0), index(1), icomp) = point(icomp);
                });
              }
            });
      });

  Kokkos::fence();

  auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result_view);

  for (int i = 0; i < n_edges; ++i) {
    for (int j = 0; j < TransferFunction2D::nquad_intersection; ++j) {
      if (!specfem::utilities::is_close(result_host(i, j, 0),
                                        expected[i][j][0])) {
        std::ostringstream oss;
        oss << "-- Transfer function --\n"
            << TransferFunction2D::description() << std::endl
            << "-- Edge Function --\n"
            << EdgeFunction2D::description() << std::endl
            << "\n-- Failure --\n"
            << "Transfer function test failed at edge " << i << ": expected "
            << expected[i][j][0] << "\n got " << result_host(i, j, 0)
            << std::endl;

        ADD_FAILURE() << oss.str();
      }
    }
  }
}

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes>
struct TransferFunctionTest2D : public ::testing::Test {
  using TransferFunctionInitializer = std::tuple_element_t<0, TestingTypes>;
  using FunctionInitializer = std::tuple_element_t<1, TestingTypes>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        function(FunctionInitializer()) {}

  specfem::test_fixture::TransferFunction2D<TransferFunctionInitializer>
      transfer_function;                                               /**< Test
                                                  transfer
                                                  function
                                                */
  specfem::test_fixture::EdgeFunction2D<FunctionInitializer> function; /**<
                                                                           Test
                                                                           field
                                                                         */
};
using namespace specfem::test_fixture;

/** Test type combinations for parameterized testing */
using TransferFunctionTestTypes2D = ::testing::Types<
    std::tuple<specfem::test_fixture::TransferFunctionInitializer2D::Zero,
               specfem::test_fixture::EdgeFunctionInitializer2D::Uniform>,
    std::tuple<TransferFunctionInitializer2D::FromQuadratureRules<
                   QuadraturePoints::GLL1, QuadraturePoints::GLL2>,
               EdgeFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Power<0>, QuadraturePoints::GLL1> >,
    std::tuple<TransferFunctionInitializer2D::FromQuadratureRules<
                   QuadraturePoints::GLL2, QuadraturePoints::GLL1>,
               EdgeFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Power<1>, QuadraturePoints::GLL2> >,
    std::tuple<TransferFunctionInitializer2D::FromQuadratureRules<
                   QuadraturePoints::GLL2, QuadraturePoints::GLL2>,
               EdgeFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Power<2>, QuadraturePoints::GLL2> >,
    std::tuple<
        TransferFunctionInitializer2D::FromQuadratureRules<
            QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
        EdgeFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Power<3>, QuadraturePoints::Asymm4Point> >,
    std::tuple<
        TransferFunctionInitializer2D::FromQuadratureRules<
            QuadraturePoints::Asymm5Point, QuadraturePoints::Asymm4Point>,
        EdgeFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Power<4>, QuadraturePoints::Asymm5Point> >,
    std::tuple<
        TransferFunctionInitializer2D::FromQuadratureRules<
            QuadraturePoints::Asymm5Point, QuadraturePoints::Asymm5Point>,
        EdgeFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Power<5>,
            QuadraturePoints::Asymm5Point> > >;

/* for test naming
 * https://google.github.io/googletest/reference/testing.html#TYPED_TEST_SUITE
 */
struct TransferFunctionTest2DNames {
  template <typename TestingTypes> static std::string GetName(int) {
    using TestType = TransferFunctionTest2D<TestingTypes>;
    return std::string("TransferFunctionTest2D(") +
           TestType::TransferFunctionInitializer::name() + ", " +
           TestType::FunctionInitializer::name() + ")";
  }
};

TYPED_TEST_SUITE(TransferFunctionTest2D, TransferFunctionTestTypes2D,
                 TransferFunctionTest2DNames);

TYPED_TEST(TransferFunctionTest2D, ExecuteTransferFunction) {
  execute(this->transfer_function, this->function);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SPECFEMEnvironment);
  return RUN_ALL_TESTS();
}
