#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>
#include <gmp.h>
#include <iostream>

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

static void mulmod_cpu_template(const Tensor& a, const Tensor& b, const Tensor& mod, Tensor& c) {
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same scalar type");
  TORCH_CHECK(a.scalar_type() == mod.scalar_type(), "a and mod must have the same scalar type");
  TORCH_CHECK(mod.dim() == 1, "mod must be a 1D tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same sizes");

  AT_DISPATCH_FIELD_TYPES(a.scalar_type(), "mulmod_cpu", [&] {
    auto a_ptr = a.const_data_ptr<scalar_t>();
    auto b_ptr = b.const_data_ptr<scalar_t>();
    auto mod_ptr = mod.const_data_ptr<scalar_t>();
    auto c_ptr = c.mutable_data_ptr<scalar_t>();
    int64_t big_int_count = a.size(a.dim()-1);
    int64_t num_ = a.numel() / big_int_count;

    mpz_t op1, op2, mod, op3;
    mpz_init(op1);
    mpz_init(op2);
    mpz_init(mod);
    mpz_init(op3);

    mpz_import(mod, big_int_count, 1, sizeof(uint64_t), 0, 0, mod_ptr);

    for(auto i = 0; i < num_; i++) {
      mpz_import(op1, big_int_count, 1, sizeof(uint64_t), 0, 0, a_ptr+i * big_int_count);
      mpz_import(op2, big_int_count, 1, sizeof(uint64_t), 0, 0, b_ptr+i * big_int_count);
      mpz_mul(op3, op1, op2);   // Multiply op1 and op2
      mpz_mod(op3, op3, mod);
      mpz_export(c_ptr+i * big_int_count, nullptr, 1, sizeof(uint32_t), 0, 0, op3);
    }
  });
}

} // namespace

Tensor mulmod_cpu(const Tensor& a, const Tensor& b, const Tensor& mod) {
  Tensor c = at::empty_like(a);
  mulmod_cpu_template(a, b, mod, c);
  return c;
}


} // namespace native
} // namespace at
