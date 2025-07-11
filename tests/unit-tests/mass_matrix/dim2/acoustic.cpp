#include "medium/compute_mass_matrix.hpp"
#include "specfem/point.hpp"
#include <gtest/gtest.h>
#include <sstream>

constexpr auto dimension = specfem::dimension::type::dim2;
constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
constexpr auto property_tag = specfem::element::property_tag::isotropic;

using PointPartialDerivativesType =
    specfem::point::partial_derivatives<dimension, true, false>;

using PointPropertiesType =
    specfem::point::properties<dimension, medium_tag, property_tag, false>;

using PointMassMatrixType = specfem::point::field<dimension, medium_tag, false,
                                                  false, false, true, false>;

TEST(MassMatrix, AcousticIsotropic2D) {

  const type_real kappa = 10.0;

  const PointPropertiesType properties(0.0, kappa);

  const PointMassMatrixType mass_matrix =
      specfem::medium::mass_matrix_component(properties);

  const PointMassMatrixType expected_mass_matrix(static_cast<type_real>(1.0) /
                                                 kappa);

  std::ostringstream message;
  message << "Mass matrix is not equal to expected value: " << mass_matrix(0)
          << " != " << expected_mass_matrix(0);
  EXPECT_TRUE(mass_matrix == expected_mass_matrix);
}
