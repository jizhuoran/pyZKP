#pragma once
#include <cstdint>

#include <c10/macros/Macros.h>

namespace c10 {

enum class FieldType : int8_t {
  Base = 0,
  Finite = 1,
  Montgomery = 2,
  COMPILE_TIME_MAX_CURVE_FIELD_TYPES = 3,
};

struct alignas(8) Field64 {
  using underlying = uint64_t;
  uint64_t val_;
  Field64() = default;
  C10_HOST_DEVICE explicit Field64(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

struct alignas(8) BigInteger {
  using underlying = uint64_t;
  uint64_t val_;
  BigInteger() = default;
  C10_HOST_DEVICE explicit BigInteger(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

struct alignas(8) FiniteField {
  using underlying = uint64_t;
  uint64_t val_;
  FiniteField() = default;
  C10_HOST_DEVICE explicit FiniteField(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

struct alignas(8) EllipticCurve {
  using underlying = uint64_t;
  uint64_t val_;
  EllipticCurve() = default;
  C10_HOST_DEVICE explicit EllipticCurve(uint64_t val) : val_(val) {}
  operator uint64_t() const { return val_; }
};

template <typename T>
struct is_field : public std::false_type {};

template<>
struct is_field<c10::Field64> : public std::true_type {};

template<>
struct is_field<c10::BigInteger> : public std::true_type {};

template<>
struct is_field<c10::FiniteField> : public std::true_type {};

template<>
struct is_field<c10::EllipticCurve> : public std::true_type {};

} // namespace c10


