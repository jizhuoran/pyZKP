#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Field64.h>

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

C10_CLANG_DIAGNOSTIC_POP()

#define C10_INTERNAL_INCLUDE_FIELD_REMAINING_H
// // math functions are included in a separate file
// #include <c10/util/complex_math.h> // IWYU pragma: keep
// utilities for field types
#include <c10/util/Field_utils.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_CFIELD_REMAINING_H
