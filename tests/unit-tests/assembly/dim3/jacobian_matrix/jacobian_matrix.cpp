#include "../test_fixture.hpp"
#include "enumerations/interface.hpp"
#include "quadrature/interface.hpp"
#include "specfem/assembly.hpp"
#include "specfem/jacobian.hpp"
#include "specfem/point.hpp"
#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>
#include <array>
#include <unordered_map>
#include <vector>

#define CHECK_VIEW_ALLOCATION(view, nspec, ngllz, nglly, ngllx)                \
  ASSERT_TRUE(view.extent(0) == nspec)                                         \
      << "View extent 0 mismatch. Expected: " << nspec                         \
      << ", Got: " << view.extent(0) << std::endl;                             \
  ASSERT_TRUE(view.extent(1) == ngllz)                                         \
      << "View extent 1 mismatch. Expected: " << ngllz                         \
      << ", Got: " << view.extent(1) << std::endl;                             \
  ASSERT_TRUE(view.extent(2) == nglly)                                         \
      << "View extent 2 mismatch. Expected: " << nglly                         \
      << ", Got: " << view.extent(2) << std::endl;                             \
  ASSERT_TRUE(view.extent(3) == ngllx)                                         \
      << "View extent 3 mismatch. Expected: " << ngllx                         \
      << ", Got: " << view.extent(3) << std::endl;

namespace specfem::assembly_test {

struct TotalGLLPoints {
  int ngllx;     ///< Number of GLL points in x direction
  int nglly;     ///< Number of GLL points in y direction
  int ngllz;     ///< Number of GLL points in z direction
  int nelements; ///< Total number of elements in the mesh

  TotalGLLPoints(int ngllx, int nglly, int ngllz, int nelements)
      : ngllx(ngllx), nglly(nglly), ngllz(ngllz), nelements(nelements) {}
};

struct Element3D {
  int element_id;
  int control_nodes_per_element;

private:
  std::vector<std::array<type_real, 3> > _coordinates;

public:
  Element3D(int element_id, int control_nodes_per_element,
            const std::initializer_list<std::array<type_real, 3> > &coords)
      : element_id(element_id),
        control_nodes_per_element(control_nodes_per_element),
        _coordinates(coords) {
    int i = 0;
    for (const auto &coord : coords) {
      _coordinates[i] = coord;
      ++i;
    }
  }

  const Kokkos::View<
      specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
      Kokkos::HostSpace>
  coordinates() const {
    const int npoints = static_cast<int>(_coordinates.size());
    Kokkos::View<
        specfem::point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace>
        coord_view("coord_view", npoints);
    for (int i = 0; i < npoints; ++i) {
      coord_view(i) = { _coordinates[i][0], _coordinates[i][1],
                        _coordinates[i][2] };
    }
    return coord_view;
  }
};

struct ExpectedJacobian3D {
  constexpr static specfem::dimension::type dimension =
      specfem::dimension::type::dim3;
  TotalGLLPoints total_gll_points;
  std::vector<Element3D> elements;

  ExpectedJacobian3D(TotalGLLPoints total_gll_points,
                     const std::initializer_list<Element3D> &elements)
      : total_gll_points(total_gll_points), elements(elements) {}

  void check(const specfem::assembly::jacobian_matrix<dimension>
                 &jacobian_matrix) const {
    ASSERT_EQ(jacobian_matrix.nspec, total_gll_points.nelements)
        << "Total number of elements mismatch. "
        << "Expected: " << total_gll_points.nelements << ", "
        << "Got: " << jacobian_matrix.nspec << std::endl;
    ASSERT_EQ(jacobian_matrix.ngllx, total_gll_points.ngllx)
        << "Total number of GLL points in x direction mismatch. "
        << "Expected: " << total_gll_points.ngllx << ", "
        << "Got: " << jacobian_matrix.ngllx << std::endl;
    ASSERT_EQ(jacobian_matrix.nglly, total_gll_points.nglly)
        << "Total number of GLL points in y direction mismatch. "
        << "Expected: " << total_gll_points.nglly << ", "
        << "Got: " << jacobian_matrix.nglly << std::endl;
    ASSERT_EQ(jacobian_matrix.ngllz, total_gll_points.ngllz)
        << "Total number of GLL points in z direction mismatch. "
        << "Expected: " << total_gll_points.ngllz << ", "
        << "Got: " << jacobian_matrix.ngllz << std::endl;

    // Check that views are allocated
    const int nspec = jacobian_matrix.nspec;
    const int ngllx = jacobian_matrix.ngllx;
    const int nglly = jacobian_matrix.nglly;
    const int ngllz = jacobian_matrix.ngllz;

    // Gernerate quadrature
    const auto quadrature = []() {
      specfem::quadrature::gll::gll gll{};
      return specfem::quadrature::quadratures(gll);
    }();

    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xix, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xiy, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_xiz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_etaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_gammaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.h_jacobian, nspec, ngllz, nglly,
                          ngllx);

    CHECK_VIEW_ALLOCATION(jacobian_matrix.xix, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.xiy, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.xiz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.etaz, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammax, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammay, nspec, ngllz, nglly, ngllx);
    CHECK_VIEW_ALLOCATION(jacobian_matrix.gammaz, nspec, ngllz, nglly, ngllx);

    const auto xi = quadrature.gll.get_hxi();

    for (const auto &element : elements) {
      const int ispec = element.element_id;
      const auto expected_coord = element.coordinates();
      for (int iz = 0; iz < ngllz; ++iz) {
        for (int iy = 0; iy < nglly; ++iy) {
          for (int ix = 0; ix < ngllx; ++ix) {
            const type_real xil = xi(ix);
            const type_real etal = xi(iy);
            const type_real zetal = xi(iz);

            const auto expected_jacobian = specfem::jacobian::compute_jacobian(
                expected_coord, element.control_nodes_per_element, xil, etal,
                zetal);

            const type_real tol = 1e-6;

            const auto point_jacobian = [&]()
                -> specfem::point::jacobian_matrix<dimension, true, false> {
              specfem::point::jacobian_matrix<dimension, true, false>
                  point_jacobian;
              specfem::assembly::load_on_host(
                  specfem::point::index<dimension, false>(ispec, iz, iy, ix),
                  jacobian_matrix, point_jacobian);
              return point_jacobian;
            }();

            EXPECT_TRUE(expected_jacobian == point_jacobian)
                << "Jacobian matrix mismatch at element " << ispec
                << ", GLL point (" << ix << ", " << iy << ", " << iz << ").";
          }
        }
      }
    }

    SUCCEED()
        << "Jacobian matrix check passed for all elements and GLL points.";
  }
};

} // namespace specfem::assembly_test

using namespace specfem::assembly_test;

static const std::unordered_map<std::string, ExpectedJacobian3D>
    expected_jacobians_3d = { { "EightNodeElastic",
                                { { 5, 5, 5, 8 }, // Total GLL points: ngllx,
                                                  // nglly, ngllz, nelements
                                  { { 0,
                                      8,
                                      { { 0.0, 50000.0, 50000.0 },
                                        { 50000.0, 50000.0, 0.0 },
                                        { 50000.0, 0.0, 0.0 },
                                        { 0.0, 0.0, 50000.0 },
                                        { 0.0, 50000.0, 50000.0 },
                                        { 50000.0, 50000.0, 0.0 },
                                        { 50000.0, 0.0, 0.0 },
                                        { 0.0, 0.0, 0.0 } } } } } } };

TEST_P(Assembly3DTest, JacobianMatrix) {
  const auto &param_name = GetParam();

  if (expected_jacobians_3d.find(param_name) == expected_jacobians_3d.end()) {
    GTEST_SKIP() << "No expected jacobian matrix data available for test case '"
                 << param_name << "'.";
    return;
  }

  const auto &assembly = getAssembly();
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &expected_jacobian = expected_jacobians_3d.at(param_name);
  expected_jacobian.check(jacobian_matrix);
}
