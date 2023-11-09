#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/native/biginteger/CurveDispatch.h>

#include "CurveDef.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

template <typename T>
static void to_mont(c10::EllipticCurve* self, const int64_t num) {
  auto self_ptr = reinterpret_cast<T*>(self);
  int64_t num_ = num / (sizeof(T)/sizeof(c10::EllipticCurve));
  for(auto i = 0; i < num_; i++) {
    self_ptr[i].to();
  }
}

static void to_mont_cpu_template(Tensor& self) {

  if(self.field() == c10::FieldType::Montgomery) {
    throw std::runtime_error("Tensor is already in Montgomery form");
  }
  
  AT_DISPATCH_CURVE_TYPES(self.scalar_type(), "to_mont_cpu", [&] {
        CURVE_DISPATCH_TYPES(self.curve().type(), 
                        to_mont, 
                        self.mutable_data_ptr<scalar_t>(),
                        self.numel());
  });

  self.set_field(c10::FieldType::Montgomery);
}


template <typename T>
static void to_base(c10::EllipticCurve* self, const int64_t num) {
  auto self_ptr = reinterpret_cast<T*>(self);
  int64_t num_ = num / (sizeof(T)/sizeof(c10::EllipticCurve));
  for(auto i = 0; i < num_; i++) {
    self_ptr[i].from();
  }
}

static void to_base_cpu_template(Tensor& self) {

  if(self.field() == c10::FieldType::Base) {
    throw std::runtime_error("Tensor is already in base form");
  }
  
  AT_DISPATCH_CURVE_TYPES(self.scalar_type(), "to_base_cpu", [&] {
        CURVE_DISPATCH_TYPES(self.curve().type(), 
                        to_base, 
                        self.mutable_data_ptr<scalar_t>(),
                        self.numel());
  });

  self.set_field(c10::FieldType::Base);
}

} // namespace

Tensor to_mont_cpu(const Tensor& input) {
  Tensor output = input.clone();
  to_mont_cpu_template(output);
  return output;
}

Tensor& to_mont_cpu_(Tensor& self) {
  to_mont_cpu_template(self);
  return self;
}

Tensor& to_mont_out_cpu(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_mont_cpu_template(output);
  return output;
}

Tensor to_base_cpu(const Tensor& input) {
  Tensor output = input.clone();
  to_base_cpu_template(output);
  return output;
}

Tensor& to_base_cpu_(Tensor& self) {
  to_base_cpu_template(self);
  return self;
}

Tensor& to_base_out_cpu(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_base_cpu_template(output);
  return output;
}

} // namespace native
} // namespace at
