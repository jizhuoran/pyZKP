#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Field64.h>
#include <c10/util/BigInteger.h>

// #if defined(__CUDACC__) || defined(__HIPCC__)
// #include <thrust/complex.h>
// #endif

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif
#if C10_CLANG_HAS_WARNING("-Wfloat-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wfloat-conversion")
#endif


// Define macros to generate lambda functions that return the type name






#define GENERATE_CASES_UP_TO_MAX_BIGINT(DISPATCH_MACRO) \
    DISPATCH_MACRO(1) \
    DISPATCH_MACRO(2) \
    DISPATCH_MACRO(3) \
    DISPATCH_MACRO(4) \
    DISPATCH_MACRO(5) \
    DISPATCH_MACRO(6) \
    DISPATCH_MACRO(7) \
    DISPATCH_MACRO(8) \
    DISPATCH_MACRO(9) \
    DISPATCH_MACRO(10) \
    DISPATCH_MACRO(11) \
    DISPATCH_MACRO(12) \
    DISPATCH_MACRO(13) \
    DISPATCH_MACRO(14) \
    DISPATCH_MACRO(15) \
    DISPATCH_MACRO(16) \
    DISPATCH_MACRO(17) \
    DISPATCH_MACRO(18) \
    DISPATCH_MACRO(19) \
    DISPATCH_MACRO(20) \
    DISPATCH_MACRO(21) \
    DISPATCH_MACRO(22) \
    DISPATCH_MACRO(23) \
    DISPATCH_MACRO(24) \
    DISPATCH_MACRO(25) \
    DISPATCH_MACRO(26) \
    DISPATCH_MACRO(27) \
    DISPATCH_MACRO(28) \
    DISPATCH_MACRO(29) \
    DISPATCH_MACRO(30) \
    DISPATCH_MACRO(31) \
    DISPATCH_MACRO(32)


C10_CLANG_DIAGNOSTIC_POP()

#define C10_INTERNAL_INCLUDE_FIELD_REMAINING_H
// // math functions are included in a separate file
// #include <c10/util/complex_math.h> // IWYU pragma: keep
// utilities for field types
#include <c10/util/Field_utils.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_CFIELD_REMAINING_H
