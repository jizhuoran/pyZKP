#pragma once

#if !defined(C10_INTERNAL_INCLUDE_FIELD_REMAINING_H)
#error \
    "c10/util/Field_utils.h is not meant to be individually included. Include c10/util/Field.h instead."
#endif

#include <limits>

namespace c10 {

template <typename T>
struct is_field : public std::false_type {};

template<>
struct is_field<c10::Field64> : public std::true_type {};

template<>
struct is_field<c10::InternalBigInteger> : public std::true_type {};

#define TEMPLATE_DECLARE(N) \
    template<> \
    struct is_field<c10::BigInteger<N>> : public std::true_type {};

GENERATE_CASES_UP_TO_MAX_BIGINT(TEMPLATE_DECLARE);

#undef TEMPLATE_DECLARE

// // Extract double from std::complex<double>; is identity otherwise
// // TODO: Write in more idiomatic C++17
// template <typename T>
// struct scalar_value_type {
//   using type = T;
// };
// template <typename T>
// struct scalar_value_type<std::complex<T>> {
//   using type = T;
// };
// template <typename T>
// struct scalar_value_type<c10::complex<T>> {
//   using type = T;
// };

} // namespace c10

// namespace std {

// template <typename T>
// class numeric_limits<c10::complex<T>> : public numeric_limits<T> {};

// template <typename T>
// bool isnan(const c10::complex<T>& v) {
//   return std::isnan(v.real()) || std::isnan(v.imag());
// }

// } // namespace std
