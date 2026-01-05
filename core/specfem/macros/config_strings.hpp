#pragma once

#include "macros_impl/seq_cat_with_or.hpp"
#include <boost/preprocessor.hpp>

// ------- Implemetation Details --------

#define _TEST_CONFIG_STRING(s, data, elem) str_lower == #elem

#define _DEFINE_CONFIG_STRING_FUNCTIONS(s, data, elem)                         \
  bool BOOST_PP_CAT(BOOST_PP_CAT(is_, BOOST_PP_TUPLE_ELEM(0, elem)),           \
                    _string)(const std::string &str) {                         \
    const auto str_lower = to_lower(str);                                      \
    return _SEQ_CAT_WITH_OR(BOOST_PP_SEQ_TRANSFORM(                            \
        _TEST_CONFIG_STRING, _, BOOST_PP_TUPLE_TO_SEQ(elem)));                 \
  }

#define _DECLARE_CONFIG_STRING_FUNCTIONS(s, data, elem)                        \
  bool BOOST_PP_CAT(BOOST_PP_CAT(is_, BOOST_PP_TUPLE_ELEM(0, elem)),           \
                    _string)(const std::string &str);

// ------- Public Definitions --------

// clang-format off
#define CONFIG_STRINGS \
 ((hdf5, h5)) \
 ((adios2, bp)) \
 ((ascii, txt)) \
 ((npy, numpy)) \
 ((npz, numpy_zip)) \
 ((psv, p_sv, p-sv)) \
 ((sh)) \
 ((te)) \
 ((tm)) \
 ((jpg, jpeg)) \
 ((vtkhdf)) \
 ((png)) \
 ((sac)) \
 ((su, seismic_unix, seismic-unix)) \
 ((forward)) \
 ((combined)) \
 ((backward)) \
 ((adjoint)) \
 ((gll4, gll-4, gll_4)) \
 ((gll7, gll-7, gll_7)) \
 ((displacement, disp, displ, d)) \
 ((velocity, vel, v, veloc)) \
 ((acceleration, acc, a, accel)) \
 ((pressure, p, pres)) \
 ((rotation, r, rot)) \
 ((intrinsic_rotation, intrinsic_rot, intrinsic_r, intrinsic_rotat, intrinsic-rotation, ir)) \
 ((curl, crl)) \
 ((newmark)) \
 ((onscreen, on-screen, on_screen)) \
 ((x, xc, xcomp, x_component)) \
 ((y, yc, ycomp, y_component)) \
 ((z, zc, zcomp, z_component)) \
 ((magnitude, m, mag))
// clang-format on
