#pragma once

#if defined(SPECFEMPP_MSVC_BUILD)
// mscv (microsoft)
// this is untested.

#define __SUPPRESS_WARNING_END() __pragma("warning( pop )")

#define _SUPPRESS_TEMPORARY_RETURN_CODE C4172

// https://learn.microsoft.com/en-us/cpp/preprocessor/warning?view=msvc-170#push-and-pop
#define _SUPPRESS_WARNING_START(CODE)                                          \
  __pragma("warning( push )")                                                  \
      __pragma(STRINGIFY("warning( disable : " #CODE " )"))

#endif

#if defined(SPECFEMPP_GNU_BUILD)

#define _SUPPRESS_WARNING_END() _Pragma("GCC diagnostic pop")

#define _SUPPRESS_TEMPORARY_RETURN_CODE "-Wreturn-local-addr"

// gcc
// https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html

// clang should also be reading these (first paragraph of
// https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via-pragmas
//)
#define _SUPPRESS_WARNING_START(CODE)                                          \
  _Pragma("GCC diagnostic push")                                               \
      _Pragma(STRINGIFY("GCC diagnostic ignored " #CODE))

#endif

#if defined(SPECFEMPP_APPLE_BUILD)
// apple clang
// https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-via
// pragmas

#define _SUPPRESS_WARNING_END() _Pragma("clang diagnostic pop")

#define _SUPPRESS_TEMPORARY_RETURN_CODE()                                      \
  _Pragma("clang diagnostic ignored \"-Wreturn-stack-address\"")

#define _SUPPRESS_WARNING_START() _Pragma("clang diagnostic push")

#endif

// Fallback definitions for unsupported compilers
#if !defined(_SUPPRESS_WARNING_START) || !defined(_SUPPRESS_WARNING_END)
#define _SUPPRESS_WARNING_START(CODE)
#define _SUPPRESS_WARNING_END()
#endif

#if !defined(_SUPPRESS_TEMPORARY_RETURN_CODE)
#define _SUPPRESS_TEMPORARY_RETURN_CODE
#endif

#define SUPPRESS_TEMPORARY_REF(CODEBLOCK)                                      \
  _SUPPRESS_WARNING_START()                                                    \
  _SUPPRESS_TEMPORARY_RETURN_CODE()                                            \
  CODEBLOCK                                                                    \
  _SUPPRESS_WARNING_END()
