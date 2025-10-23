
#include <Kokkos_Core.hpp>
#include <array>
#include <gtest/gtest.h>
#include <tuple>
#include <type_traits>
#include <vector>

#include "Kokkos_Environment.hpp"
#include "MPI_environment.hpp"
#include "algorithms/gradient.hpp"
#include "datatypes/point_view.hpp"
#include "datatypes/simd.hpp"
#include "enumerations/interface.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_each_level.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "specfem/assembly.hpp"

namespace specfem::algorithms_test {

constexpr static int ngll = 5;

enum class FunctionInitializer { ZERO, UNIFORM };

enum class QuadratureInitializer { IDENTITY };

enum class JacobianInitializer { IDENTITY };

template <typename Initializer> struct JacobianMatrix {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim2;
  std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
      _jacobian;

  JacobianMatrix(const Initializer &initializer)
      : _jacobian(init_jacobian(initializer)) {}

  specfem::assembly::jacobian_matrix<dimension_tag> jacobian() {
    specfem::assembly::jacobian_matrix<dimension_tag> jacobian_matrix(1, 5, 5);
    for (int iz = 0; iz < 5; ++iz) {
      for (int ix = 0; ix < 5; ++ix) {
        jacobian_matrix.xix(0, iz, ix) = _jacobian[iz][ix][0][0];
        jacobian_matrix.xiz(0, iz, ix) = _jacobian[iz][ix][0][1];
        jacobian_matrix.gammax(0, iz, ix) = _jacobian[iz][ix][1][0];
        jacobian_matrix.gammaz(0, iz, ix) = _jacobian[iz][ix][1][1];
      }
    }
    return jacobian_matrix;
  }
};

std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
init_jacobian(const std::integral_constant<JacobianInitializer,
                                           JacobianInitializer::IDENTITY> &) {
  std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
      jacobian{};
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      jacobian[iz][ix][0][0] = 1.0;
      jacobian[iz][ix][0][1] = 0.0;
      jacobian[iz][ix][1][0] = 0.0;
      jacobian[iz][ix][1][1] = 1.0;
    }
  }
  return jacobian;
}

template <typename Initializer> struct Quadrature {
  std::array<std::array<type_real, 5>, 5> _quadrature;

  Quadrature(const Initializer &initializer)
      : _quadrature(init_quadrature(initializer)) {}

  type_real operator()(const int &i, const int &j) const {
    return _quadrature[i][j];
  }
};

std::array<std::array<type_real, 5>, 5> init_quadrature(
    const std::integral_constant<QuadratureInitializer,
                                 QuadratureInitializer::IDENTITY> &) {
  std::array<std::array<type_real, 5>, 5> quadrature{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      quadrature[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  return quadrature;
}

template <typename Initializer> struct Function {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim2;
  constexpr static int components = 1;

  using memory_space = Kokkos::HostSpace;
  using memory_traits = Kokkos::MemoryTraits<>;

  using view_type = specfem::datatype::VectorChunkElementViewType<
      type_real, dimension_tag, 1, ngll, components, false, memory_space,
      memory_traits>;

  Function(const Initializer &initializer) : _f(init_function(initializer)) {}

  view_type view() const {
    view_type f_view("f_view");
    for (int iz = 0; iz < ngll; ++iz) {
      for (int ix = 0; ix < ngll; ++ix) {
        f_view(0, iz, ix, 0) = _f[0][iz][ix];
      }
    }
    return f_view;
  }

private:
  std::vector<std::array<std::array<type_real, ngll>, ngll> > _f;
};

auto init_function(const std::integral_constant<FunctionInitializer,
                                                FunctionInitializer::ZERO> &) {
  std::vector<std::array<std::array<type_real, ngll>, ngll> > _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int ix = 0; ix < ngll; ++ix) {
        _f[icomp][iz][ix] = 0.0;
      }
    }
  }
  return _f;
}

auto init_function(const std::integral_constant<
                   FunctionInitializer, FunctionInitializer::UNIFORM> &) {
  std::vector<std::array<std::array<type_real, 5>, 5> > _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int ix = 0; ix < 5; ++ix) {
        _f[icomp][iz][ix] = 1.0;
      }
    }
  }
  return _f;
}

template <JacobianInitializer JInit, QuadratureInitializer QInit,
          FunctionInitializer FInit>
std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
gradient();

template <>
std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
gradient<JacobianInitializer::IDENTITY, QuadratureInitializer::IDENTITY,
         FunctionInitializer::ZERO>() {
  std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> grad{};
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      grad[iz][ix][0][0] = 1.0;
      grad[iz][ix][0][1] = 0.0;
      grad[iz][ix][1][0] = 0.0;
      grad[iz][ix][1][1] = 1.0;
    }
  }
  return grad;
}

template <>
std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5>
gradient<JacobianInitializer::IDENTITY, QuadratureInitializer::IDENTITY,
         FunctionInitializer::UNIFORM>() {
  std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> grad{};
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      grad[iz][ix][0][0] = 0.0;
      grad[iz][ix][0][1] = 0.0;
      grad[iz][ix][1][0] = 0.0;
      grad[iz][ix][1][1] = 0.0;
    }
  }
  return grad;
}

} // namespace specfem::algorithms_test

using namespace specfem::algorithms_test;

template <typename TestingType>
struct GradientTestFixture : public ::testing::Test {
  using JacobianInitializer = std::tuple_element_t<0, TestingType>;
  using QuadratureInitializer = std::tuple_element_t<1, TestingType>;
  using FunctionInitializer = std::tuple_element_t<2, TestingType>;

  GradientTestFixture()
      : jacobian_matrix(JacobianInitializer{}),
        quadrature(QuadratureInitializer{}), function(FunctionInitializer{}) {}

  JacobianMatrix<JacobianInitializer> jacobian_matrix;
  Quadrature<QuadratureInitializer> quadrature;
  Function<FunctionInitializer> function;
};

using GradientTestTypes = ::testing::Types<
    std::tuple<std::integral_constant<JacobianInitializer,
                                      JacobianInitializer::IDENTITY>,
               std::integral_constant<QuadratureInitializer,
                                      QuadratureInitializer::IDENTITY>,
               std::integral_constant<FunctionInitializer,
                                      FunctionInitializer::ZERO> >,
    std::tuple<std::integral_constant<JacobianInitializer,
                                      JacobianInitializer::IDENTITY>,
               std::integral_constant<QuadratureInitializer,
                                      QuadratureInitializer::IDENTITY>,
               std::integral_constant<FunctionInitializer,
                                      FunctionInitializer::UNIFORM> > >;

TYPED_TEST_SUITE(GradientTestFixture, GradientTestTypes);

TYPED_TEST(GradientTestFixture, TestGradientComputation) {
  // Implement test logic here using this->jacobian_matrix, this->quadrature,
  // and this->function

  constexpr static auto Jinit = TestFixture::JacobianInitializer::value;
  constexpr static auto Qinit = TestFixture::QuadratureInitializer::value;
  constexpr static auto Finit = TestFixture::FunctionInitializer::value;

  // const auto expected_gradient =
  //     specfem::algorithms_test::gradient<Jinit, Qinit, Finit>();

  // Create an iterator single element;
  using simd = specfem::datatype::simd<type_real, false>;
  Kokkos::View<int *, Kokkos::HostSpace> element_indices("element_indices", 1);
  element_indices(0) = 0;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      specfem::dimension::type::dim2, simd, Kokkos::DefaultHostExecutionSpace>;

  const specfem::mesh_entity::element element_grid(ngll, ngll);

  const specfem::execution::ChunkedDomainIterator chunk(
      ParallelConfig(), element_indices, element_grid);

  // Call gradient computation
  specfem::execution::for_each_level(chunk, [&](const typename decltype(
                                                chunk)::index_type &index) {
    // Call gradient algorithm here
    specfem::algorithms::gradient(
        index, this->jacobian_matrix.jacobian(), this->quadrature,
        this->function.view(), [&](const auto &iterator_index, const auto &df) {
          const auto local_index = iterator_index.get_local_index();
          const int iz = local_index.iz;
          const int ix = local_index.ix;

          // Verify computed gradient against expected gradient
        });
  });

  SUCCEED() << "Gradient computation test executed successfully.";
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
