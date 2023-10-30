#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

/**
 * quint4x2 is for un-signed 4 bit quantized Tensors that are packed to byte
 * boundary.
 */
struct alignas(8) Field64 {
  using underlying = uint64_t;
  uint64_t val_;
  Field64() = default;
  C10_HOST_DEVICE explicit Field64(uint64_t val) : val_(val) {}

  explicit operator uint64_t() const {
    return val_;
  }

};

} // namespace c10


