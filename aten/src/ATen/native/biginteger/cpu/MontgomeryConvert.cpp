#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/native/biginteger/CurveDispatch.h>

#include "CurveDef.h"
#include "c10/util/typeid.h"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

#define CONVERT_ELEM(name) \
else if (type == ScalarType::name##_Base) { return caffe2::TypeMeta::Make<name##_Mont>();} \
else if (type == ScalarType::name##_Mont) { return caffe2::TypeMeta::Make<name##_Base>();}

caffe2::TypeMeta get_corresponding_type(const ScalarType type) {
  if (false) { ; }
  APPLY_ALL_CURVE(CONVERT_ELEM)
  else {
    throw std::runtime_error("Unsupported curve type");
  }
}
#undef CONVERT_ELEM

static void to_mont_cpu_template(Tensor& self) {
  AT_DISPATCH_FR_BASE_TYPES(self.scalar_type(), "to_mont_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].to();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
}

static void to_base_cpu_template(Tensor& self) {
  AT_DISPATCH_FR_MONT_TYPES(self.scalar_type(), "to_base_cpu", [&] {
    auto self_ptr = reinterpret_cast<scalar_t::compute_type*>(self.mutable_data_ptr<scalar_t>());
    int64_t num_ = self.numel() / num_uint64(self.scalar_type());
    for(auto i = 0; i < num_; i++) {
      self_ptr[i].from();
    }
  });
  self.set_dtype(get_corresponding_type(self.scalar_type()));
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
