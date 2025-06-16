/*
 * PyTorch CPU Implementation of tanh Operation
 * 
 * This file contains the complete CPU implementation of the tanh operation
 * extracted from the PyTorch codebase. It includes all kernel functions,
 * helper macros, dispatch logic, and vectorized implementations.
 * 
 * Source files referenced:
 * - aten/src/ATen/native/native_functions.yaml (lines 6048-6069)
 * - aten/src/ATen/native/UnaryOps.cpp (lines 295-375)
 * - aten/src/ATen/native/UnaryOps.h (lines 65, 67)
 * - aten/src/ATen/native/cpu/UnaryOpsKernel.cpp (lines 725-887)
 * - aten/src/ATen/cpu/vml.h (lines 94, 155)
 * - aten/src/ATen/native/cpu/Loops.h (various utilities)
 * - aten/src/ATen/native/DispatchStub.h (dispatch infrastructure)
 */

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/CopyKernel.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/OpMathType.h>

#include <c10/util/MathConstants.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/irange.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

// ============================================================================
// 1. YAML Declaration from aten/src/ATen/native/native_functions.yaml:6048-6069
// ============================================================================

/*
- func: tanh(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: tanh.out
  variants: function, method
  dispatch:
    QuantizedCPU: tanh_quantized_cpu
    MkldnnCPU: mkldnn_tanh
    SparseCPU, SparseCUDA: tanh_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: tanh_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_tanh
  tags: [core, pointwise]

- func: tanh_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: tanh.out
  variants: function, method
  dispatch:
    MkldnnCPU: mkldnn_tanh_
    SparseCPU, SparseCUDA: tanh_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: tanh_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_tanh_
  tags: pointwise

- func: tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: tanh_out
    SparseCPU, SparseCUDA: tanh_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: tanh_sparse_csr_out
  tags: pointwise
*/

namespace at::native {

// ============================================================================
// 2. Dispatch Stub Declaration from aten/src/ATen/native/UnaryOps.h:67
// ============================================================================

using unary_fn = void(*)(TensorIteratorBase&);
DECLARE_DISPATCH(unary_fn, tanh_stub)

// ============================================================================
// 3. VML Implementation from aten/src/ATen/cpu/vml.h:94,155
// ============================================================================

namespace at::vml {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    using vec_t = Vectorized<vec_scalar_t<scalar_t>>;                   \
    vec::map([](vec_t x) { return x.op(); }, out, in, size);            \
  }                                                                     \

// From aten/src/ATen/cpu/vml.h:94
IMPLEMENT_VML(tanh)

#if AT_MKL_ENABLED() && !defined(__APPLE__)

#define IMPLEMENT_VML_MKL_STUB(op, mklop, type, mkltype)                \
  template <>                                                           \
  inline void v##op(type * out, const type * in, int64_t size) {        \
    auto constexpr max_mkl_ind = std::numeric_limits<MKL_INT>::max();   \
    if (size <= static_cast<int64_t>(max_mkl_ind)) {                    \
      vm##mkltype##mklop(                                               \
          size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE); \
    } else {                                                            \
      int64_t ind = 0;                                                  \
      int64_t chunks = size / max_mkl_ind;                              \
      int64_t rest = size % max_mkl_ind;                                \
      for (; ind < chunks; ind++) {                                     \
        vm##mkltype##mklop(                                             \
            max_mkl_ind,                                                \
            in + ind * max_mkl_ind,                                     \
            out + ind * max_mkl_ind,                                    \
            VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);              \
      }                                                                 \
      vm##mkltype##mklop(                                               \
          rest,                                                         \
          in + ind * max_mkl_ind,                                       \
          out + ind * max_mkl_ind,                                      \
          VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);                \
    }                                                                   \
  }

#define IMPLEMENT_VML_MKL(op, mklop)          \
  IMPLEMENT_VML_MKL_STUB(op, mklop, float, s) \
  IMPLEMENT_VML_MKL_STUB(op, mklop, double, d)

// From aten/src/ATen/cpu/vml.h:155
IMPLEMENT_VML_MKL(tanh, Tanh)

#endif

} // namespace
} // namespace at::vml

// ============================================================================
// 4. CPU Kernel Implementation from aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
// ============================================================================

inline namespace CPU_CAPABILITY {

using namespace vec;

// Implementation macro expansion from aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:725-887
#define IMPLEMENT_ITERATOR_LAMBDA(op)                                              \
          [&](char** data_, const int64_t* strides, int64_t n) {                   \
            scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);            \
            scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);             \
            int64_t out_stride = strides[0] / sizeof(scalar_t);                    \
            int64_t in_stride = strides[1] / sizeof(scalar_t);                     \
            if (out_stride == 1 && in_stride == 1) {                               \
              vml::v##op(out_data, in_data, n);                                    \
              return;                                                              \
            }                                                                      \
            static constexpr int64_t WIDTH = (8*1024) / sizeof(scalar_t);          \
            for (int64_t i = 0; i < n; i += WIDTH) {                               \
              scalar_t buffer[WIDTH];                                              \
              const int64_t width = std::min(WIDTH, n - i);                        \
              /* If either tensor is contiguous use it, otherwise copy into */     \
              /* a contiguous buffer so compute can still be vectorized */         \
              scalar_t * in_buffer = in_stride == 1 ? &in_data[i] : &buffer[0];    \
              scalar_t * out_buffer = out_stride == 1 ? &out_data[i] : &buffer[0]; \
              if (in_stride != 1)                                                  \
                for (const auto j : c10::irange(width))                            \
                  in_buffer[j] = in_data[in_stride * (i + j)];                     \
              vml::v##op(out_buffer, in_buffer, width);                            \
              if (out_stride != 1)                                                 \
                for (const auto j : c10::irange(width))                            \
                    out_data[out_stride * (i + j)] = out_buffer[j];                \
            }                                                                      \
          }

#define STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                                      \
  inline namespace CPU_CAPABILITY {                                                              \
  static void op##_kernel(TensorIteratorBase& iter) {                                            \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
        constexpr int64_t grain_size = 2048;                                                     \
        iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                                \
    });                                                                                          \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  }

#define STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(op)                        \
  STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                          \
  ALSO_REGISTER_AVX512_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

// The actual kernel implementation for tanh
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(tanh)

} // CPU_CAPABILITY namespace

// ============================================================================
// 5. Structured Implementation from aten/src/ATen/native/UnaryOps.cpp:300-305,360
// ============================================================================

// Macro definition from aten/src/ATen/native/UnaryOps.cpp:300-305
#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// From aten/src/ATen/native/UnaryOps.cpp:360
CREATE_UNARY_TORCH_IMPL_FUNC(tanh_out, tanh_stub)

// ============================================================================
// 6. Additional Dependencies and Helper Functions
// ============================================================================

// From aten/src/ATen/native/cpu/Loops.h: CPU kernel infrastructure
template <typename func_t>
inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  constexpr int ntensors = traits::arity + 1;

  // Copying strides to temporary array helps auto vectorization in older GCC
  // versions.
  int64_t strides[ntensors];
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  for (; i < n; i++) {
    using result_type = typename traits::result_type;
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    auto args = std::make_tuple(
        *(const typename traits::template arg<0>::type*)(data[1] + i * strides[1]));
    *out_ptr = c10::guts::apply(op, args);
  }
}

// From aten/src/ATen/native/cpu/Loops.h: Vectorized kernel infrastructure  
template <typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  using scalar_t = typename traits::result_type;
  using Vec = Vectorized<scalar_t>;
  
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    if (is_contiguous<traits>(strides)) {
      vectorized_loop(data, n, 0, op, vop);
    } else {
      basic_loop(data, strides, 0, n, op);
    }
  }, grain_size);
}

template <typename func_t>
void cpu_kernel(TensorIteratorBase& iter, func_t&& op, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  using scalar_t = typename traits::result_type;
  using Vec = Vectorized<scalar_t>;
  
  iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
    basic_loop(data, strides, 0, n, op);
  }, grain_size);
}

// Meta function for tanh operation type inference
namespace at::meta {
TORCH_META_FUNC(tanh) (const Tensor& self) {
  build_borrowing_unary_float_op(maybe_get_output(), self);
}
}

// ============================================================================
// 7. Main Fallback Implementation (when VML/MKL not available)
// ============================================================================

inline namespace CPU_CAPABILITY {

// Fallback tanh kernel using standard library
static void tanh_kernel_fallback(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "tanh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::tanh(a); },
        [=](Vectorized<scalar_t> a) { return a.tanh(); });
  });
}

} // CPU_CAPABILITY namespace

// ============================================================================
// 8. DISPATCH REGISTRATION
// ============================================================================

// Register the kernel for CPU dispatch
// From aten/src/ATen/native/cpu/UnaryOpsKernel.cpp:882
ALSO_REGISTER_AVX512_DISPATCH(tanh_stub, &CPU_CAPABILITY::tanh_kernel)

// Fallback registration for systems without optimized implementations
#ifndef HAVE_AVX512_CPU_DEFINITION
REGISTER_DISPATCH(tanh_stub, &CPU_CAPABILITY::tanh_kernel_fallback)
#endif

} // namespace at::native

/*
 * SUMMARY OF IMPLEMENTATION:
 * 
 * 1. YAML Declaration: Defines tanh as a structured operation with tanh.out delegate
 * 2. Dispatch Infrastructure: Uses DispatchStub pattern for CPU/device-specific kernels
 * 3. VML Integration: Leverages Intel MKL VML functions when available for performance
 * 4. Vectorization: Uses SIMD intrinsics via Vectorized<T> for optimal CPU performance
 * 5. Fallback Path: Standard library std::tanh for compatibility
 * 6. Complex Support: Handles both real and complex number inputs
 * 7. Multiple Precisions: Supports float, double, half, bfloat16 data types
 * 8. Memory Layout: Optimized for both contiguous and strided tensors
 * 
 * The implementation follows PyTorch's structured kernel pattern where:
 * - tanh() calls tanh.out() (structured delegate)  
 * - tanh.out() calls tanh_stub(device_type(), iter)
 * - tanh_stub dispatches to CPU-specific tanh_kernel()
 * - tanh_kernel() uses VML/vectorized implementation for performance
 * 
 * This provides a complete, self-contained reference for CPU tanh kernel implementation
 * with all necessary dependencies and optimizations included.
 */