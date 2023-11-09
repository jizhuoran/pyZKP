#pragma once

# include "mont_t.cuh"

namespace at { 
namespace native {

#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)

static __device__ __constant__ __align__(16) const uint32_t BLS12_377_P[12] = {
    TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
    TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
    TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_377_RR[12] = { /* (1<<768)%P */
    TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
    TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
    TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_377_one[12] = { /* (1<<384)%P */
    TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
    TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
    TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_381_Px128[12] = { /* left-aligned value of the modulus */
    TO_CUDA_T(0x8460000000000080), TO_CUDA_T(0x85aea21800000042),
    TO_CUDA_T(0x79b117dd04a4000b), TO_CUDA_T(0x116cf9807a89c78f),
    TO_CUDA_T(0x1d82e03650a49d8d), TO_CUDA_T(0xd71d230be2887563)
};
static __device__ __constant__ /*const*/ uint32_t BLS12_377_M0 = 0xffffffff;

typedef mont_t<253, device::BLS12_377_r, device::BLS12_377_m0,
                    device::BLS12_377_rRR, device::BLS12_377_rone,
                    device::BLS12_377_rx8> bls12_377_fr_mont;
struct BLS12_377_Fr_G1 : public bls12_377_fr_mont {
    using mem_t = BLS12_377_Fr_G1;
    __device__ __forceinline__ BLS12_377_Fr_G1() {}
    __device__ __forceinline__ BLS12_377_Fr_G1(const bls12_377_fr_mont& a) : bls12_377_fr_mont(a) {}
};


static __device__ __constant__ __align__(16) const uint32_t BLS12_377_P[12] = {
    TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
    TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
    TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_377_RR[12] = { /* (1<<768)%P */
    TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
    TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
    TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_377_one[12] = { /* (1<<384)%P */
    TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
    TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
    TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
};
static __device__ __constant__ __align__(16) const uint32_t BLS12_381_Px128[12] = { /* left-aligned value of the modulus */
    TO_CUDA_T(0x8460000000000080), TO_CUDA_T(0x85aea21800000042),
    TO_CUDA_T(0x79b117dd04a4000b), TO_CUDA_T(0x116cf9807a89c78f),
    TO_CUDA_T(0x1d82e03650a49d8d), TO_CUDA_T(0xd71d230be2887563)
};
static __device__ __constant__ /*const*/ uint32_t BLS12_377_M0 = 0xffffffff;

typedef mont_t<377, device::BLS12_377_P, device::BLS12_377_M0,
                    device::BLS12_377_RR, device::BLS12_377_one,
                    device::BLS12_381_Px128> bls12_377_fq_mont;
struct BLS12_377_Fq_G1 : public bls12_377_fq_mont {
    using mem_t = BLS12_377_Fq_G1;
    __device__ __forceinline__ BLS12_377_Fq_G1() {}
    __device__ __forceinline__ BLS12_377_Fq_G1(const bls12_377_fq_mont& a) : bls12_377_fq_mont(a) {}
};

} // namespace native
} // namespace at
