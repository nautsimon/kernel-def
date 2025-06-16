/*
 * Complete CPU Implementation of log1p operation from PyTorch codebase
 * 
 * This file contains all the CPU-specific code for the log1p operation,
 * including the main kernel function, helper functions, macros, and
 * dispatch registration logic extracted from the PyTorch repository.
 * 
 * Sources:
 * - aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
 * - aten/src/ATen/native/UnaryOps.cpp  
 * - aten/src/ATen/native/UnaryOps.h
 * - aten/src/ATen/cpu/vml.h
 * - aten/src/ATen/native/cpu/Loops.h
 * - aten/src/ATen/native/native_functions.yaml
 * - c10/util/complex_math.h
 * - aten/src/ATen/NumericUtils.h
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

namespace at::native {

// From aten/src/ATen/native/UnaryOps.h
using unary_fn = void(*)(TensorIteratorBase&);

// From aten/src/ATen/native/UnaryOps.h - line 57
DECLARE_DISPATCH(unary_fn, log1p_stub)

// From aten/src/ATen/native/UnaryOps.cpp - line 1011
DEFINE_DISPATCH(log1p_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// From aten/src/ATen/native/UnaryOps.cpp - lines 300-304
#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// From aten/src/ATen/native/UnaryOps.cpp - line 341
CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)

// From aten/src/ATen/NumericUtils.h - lines 167-180
template <typename T>
C10_HOST_DEVICE inline T log1p(T x) {
  static_assert(
      !std::is_same_v<T, double>,
      "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
  // use __logf fast approximation for peak bandwidth
  // NOTE: There is no __log1pf so unfortunately we lose precision.
  return __logf(1.0f + x);
#else
  return ::log1p(x);
#endif
}

template <>
C10_HOST_DEVICE inline double log1p<double>(double x) {
  return ::log1p(x);
}

// From c10/util/complex_math.h - lines 293-342
template <typename T>
inline c10::complex<T> log1p(const c10::complex<T>& z) {
#if defined(__APPLE__) || defined(__MACOSX) || defined(__CUDACC__) || \
    defined(__HIPCC__)
  // For Mac, the new implementation yielded a high relative error. Falling back
  // to the old version for now.
  // See https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  // For CUDA we also use this one, as thrust::log(thrust::complex) takes
  // *forever* to compile

  // log1p(z) = log(1 + z)
  // Let's define 1 + z = r * e ^ (i * a), then we have
  // log(r * e ^ (i * a)) = log(r) + i * a
  // With z = x + iy, the term r can be written as
  // r = ((1 + x) ^ 2 + y ^ 2) ^ 0.5
  //   = (1 + x ^ 2 + 2 * x + y ^ 2) ^ 0.5
  // So, log(r) is
  // log(r) = 0.5 * log(1 + x ^ 2 + 2 * x + y ^ 2)
  //        = 0.5 * log1p(x * (x + 2) + y ^ 2)
  // we need to use the expression only on certain condition to avoid overflow
  // and underflow from `(x * (x + 2) + y ^ 2)`
  T x = z.real();
  T y = z.imag();
  T zabs = std::abs(z);
  T theta = std::atan2(y, x + T(1));
  if (zabs < 0.5) {
    T r = x * (T(2) + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {T(0.5) * std::log1p(r), theta};
  } else {
    T z0 = std::hypot(x + 1, y);
    return {std::log(z0), theta};
  }
#else
  // CPU path
  // Based on https://github.com/numpy/numpy/pull/22611#issuecomment-1667945354
  c10::complex<T> u = z + T(1);
  if (u == T(1)) {
    return z;
  } else {
    auto log_u = log(u);
    if (u - T(1) == z) {
      return log_u;
    }
    return log_u * (z / (u - T(1)));
  }
#endif
}

// From aten/src/ATen/cpu/vml.h - lines 40-90
namespace vml {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    using vec_t = Vectorized<vec_scalar_t<scalar_t>>;                   \
    vec::map([](vec_t x) { return x.op(); }, out, in, size);            \
  }                                                                     \

IMPLEMENT_VML(log1p)

#if AT_MKL_ENABLED() && !defined(__APPLE__)

// NB: LP64 MKL is the most commonly used and thus we assume it here. That means
// we need to expect MKL_INT to be of type int, which implies int32_t or int64_t in most
// cases.
static_assert(
    std::is_same_v<MKL_INT, int32_t> || std::is_same_v<MKL_INT, int64_t>,
    "MKL_INT is assumed to be int32_t or int64_t");

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

// Note: log1p is not vectorized in MKL version tested, so we fall back to default implementation
// IMPLEMENT_VML_MKL(log1p, Log1p)

#endif

} // namespace CPU_CAPABILITY
} // namespace vml

// From aten/src/ATen/native/cpu/UnaryOpsKernel.cpp - CPU kernel implementation
inline namespace CPU_CAPABILITY {

using namespace vec;

// From aten/src/ATen/native/cpu/UnaryOpsKernel.cpp - lines 770-780
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

// From aten/src/ATen/native/cpu/UnaryOpsKernel.cpp - lines 820-830
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

// Generate the log1p kernel
STATIC_IMPLEMENT_COMPLEX_KERNEL(log1p)

// From aten/src/ATen/native/cpu/UnaryOpsKernel.cpp - lines 860-870
#define STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(op)                        \
  STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                          \
  ALSO_REGISTER_AVX512_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

// Register log1p with AVX512 support
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log1p)

} // CPU_CAPABILITY namespace

} // namespace at::native

// From aten/src/ATen/native/native_functions.yaml - lines 3531-3548
/*
YAML Configuration for log1p operation:

- func: log1p(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: log1p.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: log1p_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: log1p_sparse_csr
  tags: [core, pointwise]

- func: log1p_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured_delegate: log1p.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: log1p_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: log1p_sparse_csr_
  tags: pointwise

- func: log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA, MPS: log1p_out
    SparseCPU, SparseCUDA: log1p_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: log1p_sparse_csr_out
  tags: pointwise
*/

/*
 * Summary of CPU Implementation Components:
 * 
 * 1. log1p_stub: Dispatch stub declared and defined for CPU routing
 * 2. CREATE_UNARY_TORCH_IMPL_FUNC: Macro that creates structured delegate implementation
 * 3. log1p_out: Structured implementation that calls the CPU kernel via stub
 * 4. log1p_kernel: CPU kernel that uses VML for vectorized computation
 * 5. VML implementation: Vector math library with SIMD optimizations
 * 6. Complex number support: Specialized log1p for complex types  
 * 7. MKL integration: Optional Intel MKL acceleration when available
 * 8. AVX512 support: Optional AVX512 vectorization registration
 * 
 * Execution Flow:
 * torch.log1p() -> log1p_out (structured delegate) -> log1p_stub -> log1p_kernel -> VML/SIMD
 */