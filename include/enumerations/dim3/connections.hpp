#pragma once

#include "../mesh_entities.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace specfem::connections {

template <> class connection_mapping<specfem::dimension::type::dim3> {
private:
  using ElementIndexView =
      Kokkos::View<int *, Kokkos::LayoutStride, Kokkos::HostSpace>;

public:
  constexpr static auto dimension_tag = specfem::dimension::type::dim3;

  connection_mapping(const int ngllz, const int nglly, const int ngllx,
                     ElementIndexView element1, ElementIndexView element2)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), element1(element1),
        element2(element2) {
    if (element1.extent(0) != element2.extent(0)) {
      throw std::runtime_error(
          "The 2 elements must have the same number of control nodes.");
    }
  }

  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to, const int iz,
                  const int iy, const int ix) const;

  std::tuple<int, int, int>
  map_coordinates(const specfem::mesh_entity::dim3::type &from,
                  const specfem::mesh_entity::dim3::type &to) const;

private:
  int ngllz;
  int nglly;
  int ngllx;

  ElementIndexView element1; ///< Control node indices for element 1
  ElementIndexView element2; ///< Control node indices for element 2
};

} // namespace specfem::connections
