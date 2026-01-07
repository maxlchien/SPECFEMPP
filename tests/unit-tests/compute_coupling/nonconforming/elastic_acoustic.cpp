#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

template <typename TestingTypes>
using ComputeCouplingTest2D_ElasticAcoustic_Natural = ComputeCouplingTestSuite<
    specfem::interface::interface_tag::elastic_acoustic,
    specfem::interface::flux_scheme_tag::natural>::
    ComputeCouplingTest2D<TestingTypes>;

using namespace specfem::test_fixture;

using TransferFunctionTestTypes2D_ElasticAcoustic_Natural = ::testing::Types<
    std::tuple<TransferFunctionInitializer2D::FromQuadratureRules<
                   QuadraturePoints::GLL2, QuadraturePoints::GLL2>,
               IntersectionFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Vectorized<
                       AnalyticalFunctionType::Power<0>,
                       AnalyticalFunctionType::Power<1> >,
                   QuadraturePoints::GLL2>,
               EdgeFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Power<0>, QuadraturePoints::GLL2>,
               IntersectionFunctionInitializer2D::FromAnalyticalFunction<
                   AnalyticalFunctionType::Vectorized<
                       AnalyticalFunctionType::Power<0>,
                       AnalyticalFunctionType::Power<1> >,
                   QuadraturePoints::GLL2> >,
    std::tuple<
        TransferFunctionInitializer2D::FromQuadratureRules<
            QuadraturePoints::Asymm4Point, QuadraturePoints::Asymm5Point>,
        IntersectionFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Vectorized<
                AnalyticalFunctionType::Power<3>,
                AnalyticalFunctionType::Power<2> >,
            QuadraturePoints::Asymm5Point>,
        EdgeFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Power<1>, QuadraturePoints::Asymm4Point>,
        IntersectionFunctionInitializer2D::FromAnalyticalFunction<
            AnalyticalFunctionType::Vectorized<
                AnalyticalFunctionType::Power<4>,
                AnalyticalFunctionType::Power<3> >,
            QuadraturePoints::Asymm5Point> > >;

TYPED_TEST_SUITE(ComputeCouplingTest2D_ElasticAcoustic_Natural,
                 TransferFunctionTestTypes2D_ElasticAcoustic_Natural);

TYPED_TEST(ComputeCouplingTest2D_ElasticAcoustic_Natural,
           ExecuteImplComputeCoupling) {
  this->run_test();
}
