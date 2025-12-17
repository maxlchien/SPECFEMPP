#pragma once

#include "enumerations/interface.hpp"
#include "specfem_setup.hpp"

#include "medium/dim2/acoustic/isotropic/properties.hpp"
#include "medium/dim2/elastic/anisotropic/properties.hpp"
#include "medium/dim2/elastic/isotropic/properties.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/properties.hpp"
#include "medium/dim2/electromagnetic/isotropic/properties.hpp"
#include "medium/dim2/poroelastic/isotropic/properties.hpp"

namespace specfem {
namespace point {

namespace impl {
namespace properties {
} // namespace properties
} // namespace impl

/**
 * @brief Properties of a quadrature point in a 2D medium
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties : impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD> {

  using base_type = impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
};

} // namespace point
} // namespace specfem
