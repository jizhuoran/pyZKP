#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * Field64 is a 64-bit value that belong to a finite field.
 */
struct alignas(8) Bls12_381 {
  using underlying = uint64_t;
  uint64_t val_;
  Bls12_381() = default;
  C10_HOST_DEVICE explicit Bls12_381(uint64_t val) : val_(val) {}

  // explicit operator uint64_t() const {
  //   return val_;
  // }

};

} // namespace c10


