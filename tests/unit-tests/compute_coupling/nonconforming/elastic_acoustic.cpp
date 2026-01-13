#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

#include <tuple>

namespace {

using namespace specfem::test_fixture;

using GLL2_Constant = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<QuadraturePoints::GLL2,
                                                       QuadraturePoints::GLL2>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<0>,
                                           AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::GLL2>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Power<0>, QuadraturePoints::GLL2>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<0>,
                                           AnalyticalFunctionType::Power<1> >,
        QuadraturePoints::GLL2> >;

using Asymm4to5_HigherOrder = std::tuple<
    TransferFunctionInitializer2D::FromQuadratureRules<
        QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<3>,
                                           AnalyticalFunctionType::Power<2> >,
        QuadraturePoints::Asymm5Point>,
    EdgeFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Power<1>, QuadraturePoints::Asymm4Point>,
    IntersectionFunctionInitializer2D::FromAnalyticalFunction<
        AnalyticalFunctionType::Vectorized<AnalyticalFunctionType::Power<4>,
                                           AnalyticalFunctionType::Power<3> >,
        QuadraturePoints::Asymm5Point> >;

template <typename TestTypes> void run_elastic_acoustic_test() {
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
      specfem::interface::interface_tag::elastic_acoustic,
      EdgeFunctionAccessor<
          specfem::interface::interface_tag::elastic_acoustic> >(
      transfer_function, intersection_normal, edge_function, expected_solution);
}

TEST(NonconformingElasticAcoustic, GLL2_Constant) {
  run_elastic_acoustic_test<GLL2_Constant>();
}

TEST(NonconformingElasticAcoustic, Asymm4to5_HigherOrder) {
  run_elastic_acoustic_test<Asymm4to5_HigherOrder>();
}

} // namespace
