#include "specfem/macros.hpp"

namespace specfem::utilities {
/**
 * @brief A constexpr function to generate a list of edges with interfaces
 * within the simulation.
 *
 * This macro uses @ref EDGES to generate a list of edges automatically.
 *
 * @return constexpr auto list of edges
 */
template <specfem::dimension::type DimensionTag> constexpr auto edges();

/**
 * @brief 2D specialization of the edges function
 *
 * @return constexpr auto list of edges for 2D
 */
template <> constexpr auto edges<specfem::dimension::type::dim2>() {
  constexpr int total_edges = BOOST_PP_SEQ_SIZE(EDGES);
  constexpr std::array<
      std::tuple<specfem::dimension::type, specfem::connections::type,
                 specfem::interface::interface_tag,
                 specfem::element::boundary_tag>,
      total_edges>
      edges{ _MAKE_CONSTEXPR_ARRAY(EDGES) };
  return edges;
}

} // namespace specfem::utilities
