#pragma once

#include "medium/dim2/acoustic/isotropic/kernels.hpp"
#include "medium/dim2/elastic/anisotropic/kernels.hpp"
#include "medium/dim2/elastic/isotropic/kernels.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/kernels.hpp"
#include "medium/dim2/electromagnetic/isotropic/kernels.hpp"
#include "medium/dim2/poroelastic/isotropic/kernels.hpp"

namespace specfem::point {

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct kernels : public impl::kernels::data_container<DimensionTag, MediumTag,
                                                      PropertyTag, UseSIMD> {
  using base_type = impl::kernels::data_container<DimensionTag, MediumTag,
                                                  PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
  constexpr static auto dimension_tag =
      DimensionTag;                                 ///< dimension of the medium
  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
};

} // namespace specfem::point
