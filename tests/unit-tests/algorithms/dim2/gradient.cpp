
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
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"

namespace specfem::algorithms_test {

constexpr static int ngll = 5;

namespace FunctionInitializer {
struct ZERO {};
struct UNIFORM {};
struct RANDOM {};
struct TWO_ELEMENT {};
} // namespace FunctionInitializer

namespace QuadratureInitializer {
struct IDENTITY {};
struct LAGRANGE {};
} // namespace QuadratureInitializer

namespace JacobianInitializer {
struct IDENTITY {};
struct TWO_ELEMENT {};
} // namespace JacobianInitializer

std::vector<
    std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> >
init_jacobian(const JacobianInitializer::IDENTITY &) {
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> >
      jacobian(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      jacobian[0][iz][ix][0][0] = 1.0;
      jacobian[0][iz][ix][0][1] = 0.0;
      jacobian[0][iz][ix][1][0] = 0.0;
      jacobian[0][iz][ix][1][1] = 1.0;
    }
  }
  return jacobian;
}

std::vector<
    std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> >
init_jacobian(const JacobianInitializer::TWO_ELEMENT &) {
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> >
      jacobian(2);
  for (int ielement = 0; ielement < 2; ++ielement) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int ix = 0; ix < 5; ++ix) {
        jacobian[ielement][iz][ix][0][0] = 1.0 + ielement;
        jacobian[ielement][iz][ix][0][1] = 0.0;
        jacobian[ielement][iz][ix][1][0] = 0.0;
        jacobian[ielement][iz][ix][1][1] = 1.0 + ielement;
      }
    }
  }
  return jacobian;
}

template <typename Initializer> struct JacobianMatrix {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim2;
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 2>, 5>, 5> >
      _jacobian;

  JacobianMatrix(const Initializer &initializer)
      : _jacobian(init_jacobian(initializer)) {}

  const type_real &operator()(const int ielement, const int iz, const int ix,
                              const int idim, const int jdim) const {
    return _jacobian[ielement][iz][ix][idim][jdim];
  }

  const int n_elements() const { return static_cast<int>(_jacobian.size()); }

  specfem::assembly::jacobian_matrix<dimension_tag> jacobian() {
    specfem::assembly::jacobian_matrix<dimension_tag> jacobian_matrix(
        n_elements(), 5, 5);

    for (int ielement = 0; ielement < n_elements(); ++ielement) {
      for (int iz = 0; iz < 5; ++iz) {
        for (int ix = 0; ix < 5; ++ix) {
          jacobian_matrix.xix(ielement, iz, ix) =
              _jacobian[ielement][iz][ix][0][0];
          jacobian_matrix.xiz(ielement, iz, ix) =
              _jacobian[ielement][iz][ix][0][1];
          jacobian_matrix.gammax(ielement, iz, ix) =
              _jacobian[ielement][iz][ix][1][0];
          jacobian_matrix.gammaz(ielement, iz, ix) =
              _jacobian[ielement][iz][ix][1][1];
        }
      }
    }
    return jacobian_matrix;
  }
};

std::array<std::array<type_real, 5>, 5>
init_quadrature(const QuadratureInitializer::IDENTITY &) {
  std::array<std::array<type_real, 5>, 5> quadrature{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      quadrature[i][j] = (i == j) ? 1.0 : 0.0;
    }
  }
  return quadrature;
}

std::array<std::array<type_real, 5>, 5>
init_quadrature(const QuadratureInitializer::LAGRANGE &) {
  std::array<std::array<type_real, 5>, 5> quadrature{};

  const specfem::quadrature::gll::gll gll{};
  const auto hprime = gll.get_hhprime();
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      quadrature[j][i] = hprime(i, j);
    }
  }
  return quadrature;
}

template <typename Initializer> struct Quadrature {
  std::array<std::array<type_real, 5>, 5> _quadrature;

  Quadrature(const Initializer &initializer)
      : _quadrature(init_quadrature(initializer)) {}

  type_real operator()(const int &i, const int &j) const {
    return _quadrature[i][j];
  }
};

auto init_function(const FunctionInitializer::ZERO &) {
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

auto init_function(const FunctionInitializer::UNIFORM &) {
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

auto init_function(const FunctionInitializer::RANDOM &) {
  std::vector<std::array<std::array<type_real, 5>, 5> > _f(1);
  for (int icomp = 0; icomp < 1; ++icomp) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int ix = 0; ix < 5; ++ix) {
        _f[icomp][iz][ix] = static_cast<type_real>(rand()) / RAND_MAX;
      }
    }
  }
  return _f;
}

auto init_function(const FunctionInitializer::TWO_ELEMENT &) {
  std::vector<std::array<std::array<type_real, 5>, 5> > _f(2);
  for (int ielement = 0; ielement < 2; ++ielement) {
    for (int iz = 0; iz < 5; ++iz) {
      for (int ix = 0; ix < 5; ++ix) {
        _f[ielement][iz][ix] = 1.0 + ielement;
      }
    }
  }
  return _f;
}

template <typename Initializer> struct Function {
  constexpr static specfem::dimension::type dimension_tag =
      specfem::dimension::type::dim2;
  constexpr static int components = 1;

  using memory_space = Kokkos::HostSpace;
  using memory_traits = Kokkos::MemoryTraits<>;

  using view_type = Kokkos::View<type_real *[ngll][ngll][components],
                                 memory_space, memory_traits>;

  Function(const Initializer &initializer) : _f(init_function(initializer)) {}

  const type_real &operator()(const int &ielement, const int &iz,
                              const int &ix) const {
    return _f[ielement][iz][ix];
  }

  int n_elements() const { return static_cast<int>(_f.size()); }

  view_type view() const {
    view_type f_view("f_view", _f.size());
    for (int ielement = 0; ielement < static_cast<int>(_f.size()); ++ielement) {
      for (int iz = 0; iz < ngll; ++iz) {
        for (int ix = 0; ix < ngll; ++ix) {
          f_view(ielement, iz, ix, 0) = _f[ielement][iz][ix];
        }
      }
    }
    return f_view;
  }

private:
  std::vector<std::array<std::array<type_real, ngll>, ngll> > _f;
};

std::vector<
    std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
gradient(const JacobianMatrix<JacobianInitializer::IDENTITY> &jacobian,
         const Quadrature<QuadratureInitializer::IDENTITY> &quadrature,
         const Function<FunctionInitializer::ZERO> &) {
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
      grad(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      grad[0][iz][ix][0][0] = 0.0;
      grad[0][iz][ix][0][1] = 0.0;
    }
  }
  return grad;
}

std::vector<
    std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
gradient(const JacobianMatrix<JacobianInitializer::IDENTITY> &jacobian,
         const Quadrature<QuadratureInitializer::IDENTITY> &quadrature,
         const Function<FunctionInitializer::UNIFORM> &) {
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
      grad(1);
  for (int iz = 0; iz < 5; ++iz) {
    for (int ix = 0; ix < 5; ++ix) {
      grad[0][iz][ix][0][0] = 1.0;
      grad[0][iz][ix][0][1] = 1.0;
    }
  }
  return grad;
}

template <typename Jacobian, typename Quadrature, typename Function>
std::vector<
    std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
gradient(const Jacobian &jacobian, const Quadrature &quadrature,
         const Function &function) {

  // Placeholder for actual gradient computation logic
  std::vector<
      std::array<std::array<std::array<std::array<type_real, 2>, 1>, 5>, 5> >
      grad(function.n_elements());

  for (int ielement = 0; ielement < function.n_elements(); ++ielement) {
    for (int iz = 0; iz < ngll; ++iz) {
      for (int ix = 0; ix < ngll; ++ix) {
        type_real df_dxi = 0.0;
        type_real df_gamma = 0.0;
        for (int l = 0; l < ngll; ++l) {
          // Compute gradient components here using f, jacobian, and quadrature
          df_dxi += quadrature(ix, l) * function(ielement, iz, l);
          df_gamma += quadrature(iz, l) * function(ielement, l, ix);
        }
        grad[ielement][iz][ix][0][0] =
            df_dxi * jacobian(ielement, iz, ix, 0, 0) +
            df_gamma * jacobian(ielement, iz, ix, 1, 0);
        grad[ielement][iz][ix][0][1] =
            df_dxi * jacobian(ielement, iz, ix, 0, 1) +
            df_gamma * jacobian(ielement, iz, ix, 1, 1);
      }
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
    std::tuple<JacobianInitializer::IDENTITY, QuadratureInitializer::IDENTITY,
               FunctionInitializer::ZERO>,
    std::tuple<JacobianInitializer::IDENTITY, QuadratureInitializer::IDENTITY,
               FunctionInitializer::UNIFORM>,
    std::tuple<JacobianInitializer::IDENTITY, QuadratureInitializer::LAGRANGE,
               FunctionInitializer::RANDOM>,
    std::tuple<JacobianInitializer::TWO_ELEMENT,
               QuadratureInitializer::IDENTITY,
               FunctionInitializer::TWO_ELEMENT> >;

TYPED_TEST_SUITE(GradientTestFixture, GradientTestTypes);

TYPED_TEST(GradientTestFixture, TestGradientComputation) {
  // Implement test logic here using this->jacobian_matrix, this->quadrature,
  // and this->function

  ASSERT_TRUE(this->function.n_elements() == this->jacobian_matrix.n_elements())
      << "Mismatch in number of elements between function and jacobian matrix";

  const auto expected_gradient = specfem::algorithms_test::gradient(
      this->jacobian_matrix, this->quadrature, this->function);

  // Create an iterator;
  using simd = specfem::datatype::simd<type_real, false>;
  Kokkos::View<int *, Kokkos::HostSpace> element_indices(
      "element_indices", this->function.n_elements());
  for (int i = 0; i < this->function.n_elements(); ++i) {
    element_indices(i) = i;
  }

  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      specfem::dimension::type::dim2, simd, Kokkos::DefaultHostExecutionSpace>;

  const specfem::mesh_entity::element element_grid(ngll, ngll);

  const specfem::execution::ChunkedDomainIterator chunk(
      ParallelConfig(), element_indices, element_grid);

  const auto jacobian = this->jacobian_matrix.jacobian();
  const auto function_view = this->function.view();

  using FunctionView = specfem::datatype::VectorChunkElementViewType<
      type_real, specfem::dimension::type::dim2, 1, ngll, 1, false,
      Kokkos::HostSpace, Kokkos::MemoryTraits<> >;

  // Call gradient computation
  specfem::execution::for_each_level(chunk, [&](const typename decltype(
                                                chunk)::index_type &index) {
    const FunctionView f(Kokkos::subview(function_view, index.get_range(),
                                         Kokkos::ALL(), Kokkos::ALL(),
                                         Kokkos::ALL()));
    // Call gradient algorithm here
    specfem::algorithms::gradient(
        index, jacobian, this->quadrature, f,
        [&](const auto &iterator_index, const auto &df) {
          const auto local_index = iterator_index.get_index();
          const int ispec = local_index.ispec;
          const int iz = local_index.iz;
          const int ix = local_index.ix;

          const auto e = expected_gradient[ispec][iz][ix];

          // Verify computed gradient against expected gradient
          if (!specfem::utilities::is_close(df(0, 0), e[0][0]) ||
              !specfem::utilities::is_close(df(0, 1), e[0][1])) {
            FAIL() << "Mismatch for element " << local_index.ispec << "\n"
                   << "    at quadrature point (" << iz << ", " << ix << ")\n"
                   << "    expected: (" << e[0][0] << ", " << e[0][1] << ")\n"
                   << "    got: (" << df(0, 0) << ", " << df(0, 1) << ")";
          }
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
