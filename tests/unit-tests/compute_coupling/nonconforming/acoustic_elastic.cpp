#include "enumerations/coupled_interface.hpp"
#include "nonconforming.hpp"
#include "utilities/include/fixture/nonconforming_interface/analytical_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/edge_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/intersection_function.hpp"
#include "utilities/include/fixture/nonconforming_interface/quadrature.hpp"

template <typename TestingTypes>
using ComputeCouplingTest2D_AcousticElastic_Natural = ComputeCouplingTestSuite<
    specfem::interface::interface_tag::acoustic_elastic,
    specfem::interface::flux_scheme_tag::natural>::
    ComputeCouplingTest2D<TestingTypes>;

using namespace specfem::test_fixture;

using TransferFunctionTestTypes2D = ::testing::Types<std::tuple<
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
        QuadraturePoints::Asymm5Point> > >;

TYPED_TEST_SUITE(ComputeCouplingTest2D_AcousticElastic_Natural,
                 TransferFunctionTestTypes2D);

TYPED_TEST(ComputeCouplingTest2D_AcousticElastic_Natural,
           ExecuteImplComputeCoupling) {
  this->run_test();
}
