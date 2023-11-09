#pragma once

#include <ATen/core/Tensor.h>

namespace at { 
namespace native {

#define CURVE_DISPATCH_SWITCH(TYPE, ...)              \
  [&] {                                               \
    const auto& the_type = TYPE;                      \
    c10::CurveType _st = the_type;                    \
    switch (_st) {                                    \
      __VA_ARGS__                                     \
      default:                                        \
        TORCH_CHECK(false, "Unsupported curve type"); \
    }                                                 \
  }()

#define CURVE_DISPATCH_CASE(enum_type, NAME, ...)  \
  case CurveType::enum_type: {                     \
    NAME<enum_type>(__VA_ARGS__);                  \
    break;                                         \
  }

#define CURVE_DISPATCH_CASE_TYPES(NAME, ...)              \
  CURVE_DISPATCH_CASE(ALT_BN128_Fr_G1, NAME, __VA_ARGS__) \
  CURVE_DISPATCH_CASE(ALT_BN128_Fq_G1, NAME, __VA_ARGS__) \
  CURVE_DISPATCH_CASE(BLS12_377_Fr_G1, NAME, __VA_ARGS__) \
  CURVE_DISPATCH_CASE(BLS12_377_Fq_G1, NAME, __VA_ARGS__)



#define CURVE_DISPATCH_TYPES(TYPE, NAME, ...) \
  CURVE_DISPATCH_SWITCH(TYPE, CURVE_DISPATCH_CASE_TYPES(NAME, __VA_ARGS__))


} // namespace native
} // namespace at
