#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * Field64 is a 64-bit value that belong to a finite field.
 */

struct alignas(8) InternalBigInteger {
  using underlying = uint64_t;
  uint64_t val_;
  InternalBigInteger() = default;
  C10_HOST_DEVICE explicit InternalBigInteger(uint64_t val) : val_(val) {}

};

template<size_t N>
struct BigInteger {
  // using underlying = uint64_t;
  uint64_t val_[N];
  BigInteger() = default;
//   C10_HOST_DEVICE explicit Field64(uint64_t val) : val_(val) {}

};

} // namespace c10


