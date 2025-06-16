// PyTorch Complete CPU Implementation of neg Operation
// This file contains all the necessary code for the neg operation on CPU
// Extracted from PyTorch codebase for reference

// ============================================================================
// File: aten/src/ATen/native/native_functions.yaml (relevant entries)
// ============================================================================
/*
- func: neg(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: neg.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: neg_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: neg_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_neg
  tags: [core, pointwise]

- func: neg_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: neg.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: neg_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: neg_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_neg_
  tags: pointwise

- func: neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: neg_out
    SparseCPU, SparseCUDA: neg_out_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: neg_sparse_csr_out
  tags: pointwise
*/

// ============================================================================
// Dependencies and Headers
// ============================================================================

// File: aten/src/ATen/native/UnaryOps.h
#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <c10/core/Scalar.h>

namespace at {
class Tensor;
class TensorBase;
struct TensorIteratorBase;
}

namespace at::native {

using unary_fn = void(*)(TensorIteratorBase&);

inline namespace CPU_CAPABILITY {
void neg_kernel(TensorIteratorBase &iter);
} // namespace CPU_CAPABILITY

DECLARE_DISPATCH(unary_fn, neg_stub)

} // namespace at::native

// ============================================================================
// Structured Implementation
// ============================================================================

// File: aten/src/ATen/native/UnaryOps.cpp
namespace at::native {

// CREATE_UNARY_TORCH_IMPL_FUNC macro expansion for neg_out
#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// This creates the structured kernel implementation:
// TORCH_IMPL_FUNC(neg_out) (const Tensor& self, const Tensor& result) {
//   neg_stub(device_type(), *this);
// }
CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)

} // namespace at::native

// ============================================================================
// CPU Kernel Loop Utilities
// ============================================================================

// File: aten/src/ATen/native/cpu/Loops.h (simplified essential parts)
namespace at::native { inline namespace CPU_CAPABILITY {

using namespace vec;

// Basic vectorized loop implementation for CPU kernels
template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  if constexpr (check_dynamic_cast) {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  }

  iter.for_each(make_vectorized_loop2d(std::forward<func_t>(op), std::forward<vec_func_t>(vop)), grain_size);
  iter.cast_outputs();
}

}} // namespace at::native::<anonymous>

// ============================================================================
// Vectorized Implementations
// ============================================================================

// File: aten/src/ATen/cpu/vec/vec256/vec256_int.h (negation implementations)
namespace at::cpu::vec {

// Negation implementations for integer types
// Defined here so we can utilize operator-
inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
  return Vectorized<int64_t>(0) - *this;
}

inline Vectorized<int32_t> Vectorized<int32_t>::neg() const {
  return Vectorized<int32_t>(0) - *this;
}

inline Vectorized<int16_t> Vectorized<int16_t>::neg() const {
  return Vectorized<int16_t>(0) - *this;
}

inline Vectorized<int8_t> Vectorized<int8_t>::neg() const {
  return Vectorized<int8_t>(0) - *this;
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::neg() const {
  return Vectorized<uint8_t>(0) - *this;
}

} // namespace at::cpu::vec

// File: aten/src/ATen/cpu/vec/vec256/vec256_float.h (float negation)
namespace at::cpu::vec {

// For floating point types, negation is typically implemented using XOR with sign bit
// This is a simplified representation - actual implementation uses SIMD intrinsics
template<>
inline Vectorized<float> Vectorized<float>::neg() const {
  // Actual implementation uses: _mm256_xor_ps(values, _mm256_set1_ps(-0.0f))
  // For reference purposes, conceptually: return Vectorized<float>(0.0f) - *this;
  return Vectorized<float>(_mm256_xor_ps(values, _mm256_set1_ps(-0.0f)));
}

template<>
inline Vectorized<double> Vectorized<double>::neg() const {
  // Actual implementation uses: _mm256_xor_pd(values, _mm256_set1_pd(-0.0))
  return Vectorized<double>(_mm256_xor_pd(values, _mm256_set1_pd(-0.0)));
}

template<>
inline Vectorized<c10::Half> Vectorized<c10::Half>::neg() const {
  // Half precision implementation
  return Vectorized<c10::Half>(_mm256_xor_si256(values, _mm256_set1_epi16(0x8000)));
}

template<>
inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::neg() const {
  // BFloat16 implementation
  return Vectorized<c10::BFloat16>(_mm256_xor_si256(values, _mm256_set1_epi16(0x8000)));
}

} // namespace at::cpu::vec

// File: aten/src/ATen/cpu/vec/vec256/vec256_complex.h (complex negation)
namespace at::cpu::vec {

template<typename T>
inline Vectorized<c10::complex<T>> Vectorized<c10::complex<T>>::neg() const {
  // For complex numbers, negate both real and imaginary parts
  auto zero = Vectorized<c10::complex<T>>(c10::complex<T>(T(0), T(0)));
  return zero - *this;
}

} // namespace at::cpu::vec

// ============================================================================
// Main CPU Kernel Implementation
// ============================================================================

// File: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/irange.h>

namespace at::native {

inline namespace CPU_CAPABILITY {

// NB: Ignores the negative bit on tensors
void neg_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -a; },
        [=](Vectorized<scalar_t> a) { return a.neg(); });
  });
}

} // namespace CPU_CAPABILITY

// ============================================================================
// Dispatch Registration
// ============================================================================

// Register the CPU implementation
REGISTER_DISPATCH(neg_stub, &CPU_CAPABILITY::neg_kernel)

} // namespace at::native

// ============================================================================
// TensorIterator Support Functions (Essential Dependencies)
// ============================================================================

// File: aten/src/ATen/native/TensorIterator.h (simplified)
namespace at {

struct TensorIteratorBase {
  // Essential interface for neg kernel
  ScalarType dtype() const;
  DeviceType device_type() const;
  int64_t numel() const;
  int ndim() const;
  IntArrayRef shape() const;
  
  // Core iteration method
  template<typename loop2d_t>
  void for_each(loop2d_t loop, int64_t grain_size = -1);
  
  void cast_outputs();
  
  // Factory method for unary operations
  static TensorIterator unary_op(Tensor& out, const Tensor& a);
};

} // namespace at

// ============================================================================
// AT_DISPATCH Macro Support
// ============================================================================

// This macro dispatches to different scalar types
#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(TYPE1, TYPE2, TYPE3, TYPE, NAME, ...) \
  [&] { \
    const auto& the_type = TYPE; \
    /* at::ScalarType _st = */ \
    at::ScalarType _st = the_type; \
    switch (_st) { \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, c10::Half, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexHalf, c10::complex<c10::Half>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexFloat, c10::complex<float>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::ComplexDouble, c10::complex<double>, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::BFloat16, c10::BFloat16, __VA_ARGS__) \
      default: \
        TORCH_CHECK(false, "\"", NAME, "\" not implemented for '", toString(_st), "'"); \
    } \
  }()

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: { \
    using scalar_t = type; \
    return __VA_ARGS__(); \
  }

// ============================================================================
// Summary of Execution Flow
// ============================================================================

/*
EXECUTION FLOW FOR neg OPERATION ON CPU:

1. User calls torch.neg(tensor) or tensor.neg()

2. This dispatches through the structured delegate to neg.out

3. The structured implementation calls:
   TORCH_IMPL_FUNC(neg_out)(const Tensor& self, const Tensor& result) {
     neg_stub(device_type(), *this);
   }

4. neg_stub dispatches to CPU_CAPABILITY::neg_kernel for CPU tensors

5. neg_kernel implementation:
   void neg_kernel(TensorIteratorBase& iter) {
     AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
       cpu_kernel_vec(
           iter,
           [=](scalar_t a) -> scalar_t { return -a; },           // Scalar lambda
           [=](Vectorized<scalar_t> a) { return a.neg(); });    // Vector lambda
     });
   }

6. cpu_kernel_vec handles the iteration and calls either:
   - Scalar version: -a for each element
   - Vectorized version: a.neg() for vector operations

7. For vectorized operations, Vectorized<T>::neg() implementations:
   - Integers: Vectorized<T>(0) - *this
   - Floats: XOR with sign bit using SIMD intrinsics
   - Complex: negate both real and imaginary components

The complete implementation supports:
- All integer types (int8, int16, int32, int64, uint8)
- All floating point types (half, bfloat16, float, double)
- Complex types (complex<half>, complex<float>, complex<double>)
- Automatic vectorization using AVX/AVX2 SIMD instructions
- Proper handling of different tensor layouts and strides
- Integration with PyTorch's TensorIterator for efficient memory access
*/ 