#pragma once

#include <boost/preprocessor.hpp>

// ------- Implementation Details --------

#define _SEQ_CAT_WITH_OR(seq)                                                  \
  BOOST_PP_CAT(_SEQ_CAT_WITH_OR_, BOOST_PP_SEQ_SIZE(seq)) seq

#define _SEQ_CAT_WITH_OR_1(x) x
#define _SEQ_CAT_WITH_OR_2(x) x || _SEQ_CAT_WITH_OR_1
#define _SEQ_CAT_WITH_OR_3(x) x || _SEQ_CAT_WITH_OR_2
#define _SEQ_CAT_WITH_OR_4(x) x || _SEQ_CAT_WITH_OR_3
#define _SEQ_CAT_WITH_OR_5(x) x || _SEQ_CAT_WITH_OR_4
#define _SEQ_CAT_WITH_OR_6(x) x || _SEQ_CAT_WITH_OR_5
#define _SEQ_CAT_WITH_OR_7(x) x || _SEQ_CAT_WITH_OR_6
#define _SEQ_CAT_WITH_OR_8(x) x || _SEQ_CAT_WITH_OR_7
#define _SEQ_CAT_WITH_OR_9(x) x || _SEQ_CAT_WITH_OR_8
#define _SEQ_CAT_WITH_OR_10(x) x || _SEQ_CAT_WITH_OR_9
#define _SEQ_CAT_WITH_OR_11(x) x || _SEQ_CAT_WITH_OR_10
#define _SEQ_CAT_WITH_OR_12(x) x || _SEQ_CAT_WITH_OR_11
#define _SEQ_CAT_WITH_OR_13(x) x || _SEQ_CAT_WITH_OR_12
#define _SEQ_CAT_WITH_OR_14(x) x || _SEQ_CAT_WITH_OR_13
#define _SEQ_CAT_WITH_OR_15(x) x || _SEQ_CAT_WITH_OR_14
#define _SEQ_CAT_WITH_OR_16(x) x || _SEQ_CAT_WITH_OR_15
