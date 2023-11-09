#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/biginteger/CurveDispatch.h>

#include "CurveDef.cuh"

#pragma clang diagnostic ignored "-Wmissing-prototypes"

namespace at {
namespace native {

namespace {

template <typename T>
__global__ void to_mont_kernel(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) {
    data[i].to();
  }
}

template <typename T>
static void to_mont(c10::EllipticCurve* self, const int64_t num) {
  auto self_ptr = reinterpret_cast<T*>(self);
  int64_t N = num / (sizeof(T)/sizeof(c10::EllipticCurve));
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + block_work_size() - 1) / block_work_size();
  auto stream = at::cuda::getCurrentCUDAStream();
  to_mont_kernel<<<grid, num_threads(), 0, stream>>>(N, self_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static void to_mont_cuda_template(Tensor& self) {

  if(self.field() == c10::FieldType::Montgomery) {
    throw std::runtime_error("Tensor is already in Montgomery form");
  }
  
  AT_DISPATCH_CURVE_TYPES(self.scalar_type(), "to_mont_cuda", [&] {
        CURVE_DISPATCH_TYPES(self.curve().type(), 
                        to_mont, 
                        self.mutable_data_ptr<scalar_t>(),
                        self.numel());
  });

  self.set_field(c10::FieldType::Montgomery);
}


// template <typename T>
// static void to_base(c10::EllipticCurve* self, const int64_t num) {
//   auto self_ptr = reinterpret_cast<T*>(self);
//   int64_t num_ = num / (sizeof(T)/sizeof(c10::EllipticCurve));
//   for(auto i = 0; i < num_; i++) {
//     self_ptr[i].from();
//   }
// }

// static void to_base_cuda_template(Tensor& self) {

//   if(self.field() == c10::FieldType::Base) {
//     throw std::runtime_error("Tensor is already in base form");
//   }
  
//   AT_DISPATCH_CURVE_TYPES(self.scalar_type(), "to_base_cuda", [&] {
//         CURVE_DISPATCH_TYPES(self.curve().type(), 
//                         to_base, 
//                         self.mutable_data_ptr<scalar_t>(),
//                         self.numel());
//   });

//   self.set_field(c10::FieldType::Base);
// }

} // namespace

Tensor to_mont_cuda(const Tensor& input) {
  Tensor output = input.clone();
  to_mont_cuda_template(output);
  return output;
}

Tensor& to_mont_cuda_(Tensor& self) {
  to_mont_cuda_template(self);
  return self;
}

Tensor& to_mont_out_cuda(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_mont_cuda_template(output);
  return output;
}

Tensor to_base_cuda(const Tensor& input) {
  Tensor output = input.clone();
  to_mont_cuda_template(output);
  return output;
}

Tensor& to_base_cuda_(Tensor& self) {
  to_mont_cuda_template(self);
  return self;
}

Tensor& to_base_out_cuda(const Tensor& input, Tensor& output) {
  copy(output, input);
  to_mont_cuda_template(output);
  return output;
}

} // namespace native
} // namespace at
