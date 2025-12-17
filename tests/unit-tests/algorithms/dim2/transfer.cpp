
#include <array>
#include <sstream>
#include <string>
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

#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_field.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature_rule.hpp"
#include "utilities/include/fixture/nonconforming_interface/transfer_function.hpp"

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

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

/**
 * @brief Test fixture for 2D transfer function algorithms.
 * @tparam TestingTypes Tuple of (TransferFunctionInitializer,
 * FunctionInitializer)
 */
template <typename TestingTypes, typename = void>
struct TransferFunctionTest2D : public ::testing::Test {};

template <typename TestingTypes>
struct TransferFunctionTest2D<
    TestingTypes,
    std::enable_if_t<
        std::is_base_of_v<
            specfem::test::fixture::TransferFunctionInitializer2D::Base,
            std::tuple_element_t<0, TestingTypes> > &&
            std::is_base_of_v<
                specfem::test::fixture::EdgeFieldInitializer2D::Base,
                std::tuple_element_t<1, TestingTypes> >,
        void> > : public ::testing::Test {
  using TransferFunctionInitializer = std::tuple_element_t<0, TestingTypes>;
  using FunctionInitializer = std::tuple_element_t<1, TestingTypes>;

  using TransferFunction2D =
      specfem::test::fixture::TransferFunction2D<TransferFunctionInitializer>;
  using EdgeField2D = specfem::test::fixture::EdgeField2D<FunctionInitializer>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        edge_field(FunctionInitializer()) {}

  TransferFunction2D transfer_function; /**< Test transfer function */
  EdgeField2D edge_field;               /**< Test field */

  std::vector<std::array<std::array<type_real, EdgeField2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
  expected_solution() const {
    constexpr int nquad_intersection = TransferFunction2D::nquad_intersection;
    constexpr int nquad_edge = TransferFunction2D::nquad_edge;
    constexpr int num_components = EdgeField2D::num_components;

    const int n_edges = TransferFunction2D::num_edges;
    std::vector<
        std::array<std::array<type_real, num_components>, nquad_intersection> >
        result_field(n_edges, std::array<std::array<type_real, num_components>,
                                         nquad_intersection>{ 0.0 });
    for (int i = 0; i < n_edges; ++i) {
      for (int j = 0; j < nquad_intersection; ++j) {
        for (int k = 0; k < num_components; ++k) {
          for (int l = 0; l < nquad_edge; ++l) {
            result_field[i][j][k] +=
                transfer_function(i, l, j) * edge_field(i, l, k);
          }
        }
      }
    }
    return result_field;
  }
};

template <typename AnalyticalFieldTransfer2DType>
struct TransferFunctionTest2D<
    AnalyticalFieldTransfer2DType,
    std::void_t<decltype(
        AnalyticalFieldTransfer2DType::evaluate_at_intersection_point)> >
    : public ::testing::Test {
  using TransferFunctionInitializer =
      typename AnalyticalFieldTransfer2DType::TransferFunctionInitializer;
  using FunctionInitializer =
      typename AnalyticalFieldTransfer2DType::EdgeFieldInitializer;

  using TransferFunction2D =
      specfem::test::fixture::TransferFunction2D<TransferFunctionInitializer>;
  using EdgeField2D = specfem::test::fixture::EdgeField2D<FunctionInitializer>;

  /**
   * @brief Set up test with initialized transfer function and field.
   */
  TransferFunctionTest2D()
      : transfer_function(TransferFunctionInitializer()),
        edge_field(FunctionInitializer()) {}

  TransferFunction2D transfer_function; /**< Test transfer function */
  EdgeField2D edge_field;               /**< Test field */

  std::vector<std::array<std::array<type_real, EdgeField2D::num_components>,
                         TransferFunction2D::nquad_intersection> >
  expected_solution() const {
    constexpr int nquad_intersection = TransferFunction2D::nquad_intersection;
    constexpr int nquad_edge = TransferFunction2D::nquad_edge;
    constexpr int num_components = EdgeField2D::num_components;

    const int n_edges = TransferFunction2D::num_edges;
    std::vector<
        std::array<std::array<type_real, num_components>, nquad_intersection> >
        result_field(n_edges, std::array<std::array<type_real, num_components>,
                                         nquad_intersection>{ 0.0 });
    for (int i = 0; i < n_edges; ++i) {
      for (int j = 0; j < nquad_intersection; ++j) {
        for (int k = 0; k < num_components; ++k) {
          result_field[i][j][k] +=
              AnalyticalFieldTransfer2DType::evaluate_at_intersection_point(
                  i, j, k);
        }
      }
    }
    return result_field;
  }
};

/**
 * @brief Execute transfer function test with validation.
 * @tparam TransferFunction2D Transfer function type
 * @tparam EdgeField2D Field type
 * @param transfer_function Transfer function data
 * @param function Input function data
 */
template <typename TransferFunctionTest2D>
void execute(const TransferFunctionTest2D &test) {
  using TransferFunction2D = decltype(test.transfer_function);
  using EdgeField2D = decltype(test.edge_field);

  auto &transfer_function = test.transfer_function;
  auto &edge_field = test.edge_field;

  constexpr int nquad_intersection = TransferFunction2D::nquad_intersection;
  constexpr int nquad_edge = TransferFunction2D::nquad_edge;
  constexpr int num_components = EdgeField2D::num_components;
  auto expected = test.expected_solution();

  const int n_edges = TransferFunction2D::num_edges;

  using TransferFunctionType = specfem::chunk_edge::impl::transfer_function<
      dimension_tag, 1, nquad_intersection, nquad_edge,
      specfem::data_access::DataClassType::transfer_function_self,
      interface_tag, boundary_tag, typename TransferFunction2D::memory_space,
      Kokkos::MemoryTraits<> >;
  using FunctionType = specfem::datatype::VectorChunkEdgeViewType<
      type_real, dimension_tag, 1, nquad_edge, num_components, false,
      typename EdgeField2D::memory_space, Kokkos::MemoryTraits<> >;

  const auto transfer_function_view = transfer_function.get_view();
  const auto function_view = edge_field.get_view();

  Kokkos::View<type_real *[nquad_intersection][num_components],
               typename EdgeField2D::memory_space>
      result_view("result_view", n_edges);

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
              for (int icomp = 0; icomp < num_components; ++icomp) {
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
    for (int j = 0; j < nquad_intersection; ++j) {

      if (!specfem::utilities::is_close(result_host(i, j, 0),
                                        expected[i][j][0])) {
        std::ostringstream oss;
        oss << "-- Transfer function --\n"
            << TransferFunction2D::description() << std::endl
            << "-- Edge Function --\n"
            << EdgeField2D::description() << std::endl
            << "\n-- Failure --\n"
            << "Transfer function test failed at edge " << i << ": expected "
            << expected[i][j][0] << "\n got " << result_host(i, j, 0)
            << std::endl;

        ADD_FAILURE() << oss.str();
      }
    }
  }
}

/** Test type combinations for parameterized testing */
using TransferFunctionTestTypes2D = ::testing::Types<
    std::tuple<specfem::test::fixture::TransferFunctionInitializer2D::Zero,
               specfem::test::fixture::EdgeFieldInitializer2D::Uniform>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<0>,
        specfem::test::fixture::QuadratureRuleInitializer::GLL1,
        specfem::test::fixture::QuadratureRuleInitializer::GLL2>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<1>,
        specfem::test::fixture::QuadratureRuleInitializer::GLL2,
        specfem::test::fixture::QuadratureRuleInitializer::GLL1>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<2>,
        specfem::test::fixture::QuadratureRuleInitializer::GLL2,
        specfem::test::fixture::QuadratureRuleInitializer::GLL2>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<3>,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm4Point,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm5Point>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<4>,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm5Point,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm4Point>,
    specfem::test::fixture::AnalyticalFieldTransfer2D<
        specfem::test::fixture::AnalyticalFieldInitializer1D::xi_to_the<5>,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm5Point,
        specfem::test::fixture::QuadratureRuleInitializer::Asymm5Point> >;

TYPED_TEST_SUITE(TransferFunctionTest2D, TransferFunctionTestTypes2D);

TYPED_TEST(TransferFunctionTest2D, ExecuteTransferFunction) { execute(*this); }

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
