#pragma once

#include <ATen/core/Tensor.h>
#include "cpu/ff/blst/blst_t.hpp"

namespace at { 
namespace native {

/* define Fr and Fq for ALT_BN128 */
static const vec256 ALT_BN128_r = {
    TO_LIMB_T(0x43e1f593f0000001), TO_LIMB_T(0x2833e84879b97091),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_rRR = {   /* (1<<512)%r */
    TO_LIMB_T(0x1bb8e645ae216da7), TO_LIMB_T(0x53fe3ab1e35c59e3),
    TO_LIMB_T(0x8c49833d53bb8085), TO_LIMB_T(0x0216d0b17f4e44a5)
};
static const vec256 ALT_BN128_rONE = {  /* (1<<256)%r */
    TO_LIMB_T(0xac96341c4ffffffb), TO_LIMB_T(0x36fc76959f60cd29),
    TO_LIMB_T(0x666ea36f7879462e), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<254, ALT_BN128_r, 0xc2e1f593efffffffu,
                        ALT_BN128_rRR, ALT_BN128_rONE> alt_bn128_fr_mont;

struct alignas(8) ALT_BN128_Fr_G1 : public alt_bn128_fr_mont {
  using mem_t = ALT_BN128_Fr_G1;
  inline ALT_BN128_Fr_G1() = default;
  inline ALT_BN128_Fr_G1(const alt_bn128_fr_mont& a) : alt_bn128_fr_mont(a) {}
};

static const vec256 ALT_BN128_P = {
    TO_LIMB_T(0x3c208c16d87cfd47), TO_LIMB_T(0x97816a916871ca8d),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_RR = {    /* (1<<512)%P */
    TO_LIMB_T(0xf32cfc5b538afa89), TO_LIMB_T(0xb5e71911d44501fb),
    TO_LIMB_T(0x47ab1eff0a417ff6), TO_LIMB_T(0x06d89f71cab8351f),
};
static const vec256 ALT_BN128_ONE = {   /* (1<<256)%P */
    TO_LIMB_T(0xd35d438dc58f0d9d), TO_LIMB_T(0x0a78eb28f5c70b3d),
    TO_LIMB_T(0x666ea36f7879462c), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<254, ALT_BN128_P, 0x87d20782e4866389u,
                        ALT_BN128_RR, ALT_BN128_ONE> alt_bn128_fp_mont;
struct ALT_BN128_Fq_G1 : public alt_bn128_fp_mont {
    using mem_t = ALT_BN128_Fq_G1;
    inline ALT_BN128_Fq_G1() = default;
    inline ALT_BN128_Fq_G1(const alt_bn128_fp_mont& a) : alt_bn128_fp_mont(a) {}
};



/* define Fr and Fq for BLS12_377 */
static const vec256 BLS12_377_r = { 
    TO_LIMB_T(0x0a11800000000001), TO_LIMB_T(0x59aa76fed0000001),
    TO_LIMB_T(0x60b44d1e5c37b001), TO_LIMB_T(0x12ab655e9a2ca556)
};
static const vec256 BLS12_377_rRR = {   /* (1<<512)%r */
    TO_LIMB_T(0x25d577bab861857b), TO_LIMB_T(0xcc2c27b58860591f),
    TO_LIMB_T(0xa7cc008fe5dc8593), TO_LIMB_T(0x011fdae7eff1c939)
};
static const vec256 BLS12_377_rONE = {  /* (1<<256)%r */
    TO_LIMB_T(0x7d1c7ffffffffff3), TO_LIMB_T(0x7257f50f6ffffff2),
    TO_LIMB_T(0x16d81575512c0fee), TO_LIMB_T(0x0d4bda322bbb9a9d)
};
typedef blst_256_t<253, BLS12_377_r, 0xa117fffffffffffu,
                        BLS12_377_rRR, BLS12_377_rONE> bls12_377_fr_mont;
struct BLS12_377_Fr_G1 : public bls12_377_fr_mont {
    using mem_t = BLS12_377_Fr_G1;
    inline BLS12_377_Fr_G1() = default;
    inline BLS12_377_Fr_G1(const bls12_377_fr_mont& a) : bls12_377_fr_mont(a) {}
};

static const vec384 BLS12_377_P = {
    TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
    TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
    TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x01ae3a4617c510ea)
};
static const vec384 BLS12_377_RR = {    /* (1<<768)%P */
    TO_LIMB_T(0xb786686c9400cd22), TO_LIMB_T(0x0329fcaab00431b1),
    TO_LIMB_T(0x22a5f11162d6b46d), TO_LIMB_T(0xbfdf7d03827dc3ac),
    TO_LIMB_T(0x837e92f041790bf9), TO_LIMB_T(0x006dfccb1e914b88)
};
static const vec384 BLS12_377_ONE = {   /* (1<<384)%P */
    TO_LIMB_T(0x02cdffffffffff68), TO_LIMB_T(0x51409f837fffffb1),
    TO_LIMB_T(0x9f7db3a98a7d3ff2), TO_LIMB_T(0x7b4e97b76e7c6305),
    TO_LIMB_T(0x4cf495bf803c84e8), TO_LIMB_T(0x008d6661e2fdf49a)
};
typedef blst_384_t<377, BLS12_377_P, 0x8508bfffffffffffu,
                        BLS12_377_RR, BLS12_377_ONE> bls12_377_fq_mont;
struct BLS12_377_Fq_G1 : public bls12_377_fq_mont {
    using mem_t = BLS12_377_Fq_G1;
    inline BLS12_377_Fq_G1() = default;
    inline BLS12_377_Fq_G1(const bls12_377_fq_mont& a) : bls12_377_fq_mont(a) {}
};



#define CURVE_DISPATCH_SWITCH(TYPE, ...)                                   \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    c10::CurveType _st = the_type;                                          \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        TORCH_CHECK(false, "Unsupported curve type");                       \
    }                                                                       \
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
