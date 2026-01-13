#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

#include <tuple>

namespace {

using namespace specfem::test_fixture;

using Asymm4to5_VectorizedPower = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<
        QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<0>,
                                           AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::Asymm5Point>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<1>,
                                           AnalyticalFunctionType::Power<2> >,
        QuadraturePoints::Asymm4Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Sum<AnalyticalFunctionType::Power<1>,
                                    AnalyticalFunctionType::Power<3> >,
        QuadraturePoints::Asymm5Point> >;

// Helper to run test with specific types
template <typename TestTypes> void run_acoustic_elastic_test() {
  using TransferFunctionInit = std::tuple_element_t<0, TestTypes>;
  using IntersectionNormalInit = std::tuple_element_t<1, TestTypes>;
  using EdgeFunctionInit = std::tuple_element_t<2, TestTypes>;
  using ExpectedSolutionInit = std::tuple_element_t<3, TestTypes>;

  specfem::test_fixture::TransferFunction2D<TransferFunctionInit>
      transfer_function{ TransferFunctionInit{} };
  specfem::test_fixture::IntersectionFunction2D<IntersectionNormalInit>
      intersection_normal{ IntersectionNormalInit{} };
  specfem::test_fixture::EdgeFunction2D<EdgeFunctionInit> edge_function{
    EdgeFunctionInit{}
  };
  specfem::test_fixture::IntersectionFunction2D<ExpectedSolutionInit>
      expected_solution{ ExpectedSolutionInit{} };

  execute_impl_compute_coupling<
      specfem::interface::interface_tag::acoustic_elastic,
      EdgeFunctionAccessor<
          specfem::interface::interface_tag::acoustic_elastic> >(
      transfer_function, intersection_normal, edge_function, expected_solution);
}

TEST(NonconformingAcousticElastic, Asymm4to5_VectorizedPower) {
  run_acoustic_elastic_test<Asymm4to5_VectorizedPower>();
}

} // anonymous namespace
