// File: aten/src/ATen/native/Activation.cpp (lines 511-519)
// Main ReLU function implementation - delegates to clamp_min
Tensor relu(const Tensor & self) {
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min(self, 0);
}

Tensor & relu_(Tensor & self) {
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min_(self, 0);
}

// File: aten/src/ATen/native/TensorCompare.cpp (lines 187-199)
// Meta function for clamp_min - handles type promotion and tensor setup
TORCH_META_FUNC(clamp_min)(const Tensor& self, const Scalar& min) {
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(
      !isComplexType(result_type), "clamp is not supported for complex types");
  TORCH_CHECK(!min.isComplex(), "clamp is not supported for complex types");
  // Floating is the highest supported
  if (!isFloatingType(result_type)) {
    auto result_type = at::native::result_type(self, min);
    TORCH_CHECK(
        (result_type == self.scalar_type() || !(maybe_get_output().defined()) ||
         !(maybe_get_output().is_same(self))),
        "result type ",
        result_type,
        " can't be cast to the desired output type ",
        self.dtype());
    build_unary_op(maybe_get_output(), self.to(result_type));
  } else {
    build_borrowing_unary_op(maybe_get_output(), self);
  }
}

// File: aten/src/ATen/native/TensorCompare.cpp (lines 875-885)
// Implementation function for clamp_min - calls the CPU kernel
TORCH_IMPL_FUNC(clamp_min_out)
(const Tensor& self, const Scalar& min, const Tensor& result) {
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    clamp_min_scalar_stub(device_type(), *this, min);
  }
}

// File: aten/src/ATen/native/cpu/TensorCompareKernel.cpp (lines 386-397)
// CPU kernel implementation for clamp_min
static void clamp_min_scalar_kernel_impl(TensorIteratorBase& iter, Scalar min_) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "clamp_min_scalar_cpu", [&]() {
    const auto min = min_.to<scalar_t>();
    const Vectorized<scalar_t> min_vec(min);
    cpu_kernel_vec(iter,
        [=](scalar_t a) -> scalar_t {
          return std::max(a, min);
        },
        [=](Vectorized<scalar_t> a) {
          return vec::clamp_min(a, min_vec);
        });
  });
}

// File: aten/src/ATen/native/cpu/TensorCompareKernel.cpp (lines 411-417)
// Dispatch registration for CPU
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl)

// File: aten/src/ATen/cpu/vec/vec_base.h (lines 992-1005)
// Base vectorized clamp_min implementation
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline clamp_min(
    const Vectorized<T>& a,
    const Vectorized<T>& min_vec) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = a[i] < min_vec[i] ? min_vec[i] : a[i];
  }
  return c;
}

// File: aten/src/ATen/native/cpu/Loops.h (lines 342-396)
// Core CPU loop infrastructure - cpu_kernel_vec function
template <typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types and other dtypes
  static_assert(
      std::is_same_v<typename traits::result_type, typename traits::template arg<0>::type>,
      "all types must be the same");
  static_assert(traits::arity == 1, "unary op");
  using scalar_t = typename traits::result_type;
  iter.for_each(VectorizedLoop2d<func_t, vec_func_t>(op, vop), grain_size);
}

// File: aten/src/ATen/native/cpu/Loops.h (lines 206-285)
// VectorizedLoop2d implementation for efficient CPU execution
template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;
  vop_t vop;
  using data_t = std::array<char*, 3>;

  VectorizedLoop2d(op_t op, vop_t vop):
    op(std::move(op)), vop(std::move(vop)) {}

  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, data.size(), data.begin());
    const int64_t *outer_strides = &strides[data.size()];
    for ([[maybe_unused]] const auto i : c10::irange(size1)) {
      vectorized_loop(data.data(), size0, 0, op, vop);
      advance(data, outer_strides);
    }
  }
};

// File: aten/src/ATen/native/cpu/Loops.h (lines 201-249)
// Core vectorized loop implementation
template <typename func_t, typename vec_func_t>
inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = vec::Vectorized<scalar_t>;
  char* C10_RESTRICT data[3] = {data_[0], data_[1], data_[2]};
  
  // Determine if vectorization is beneficial
  constexpr int64_t ntensors = 2;
  const int64_t stride0 = ntensors == 1 ? 0 : sizeof(scalar_t);
  const int64_t stride1 = sizeof(scalar_t);
  
  if (n < Vec::size() || n < 16384) {
    basic_loop(data, strides, 0, n, op);
    return;
  }
  
  int64_t i = 0;
  if (!is_contiguous<ntensors>(strides)) {
    basic_loop(data, strides, 0, n, op);
    return;
  }
  
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
    auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
    auto out1 = c10::guts::apply(vop, std::move(args1));
    auto out2 = c10::guts::apply(vop, std::move(args2));
    out1.store(data[0] + i * sizeof(scalar_t));
    out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
  }
  if (i < n) {
    int64_t strides[ntensors] = {stride0, stride1};
    basic_loop(data, strides, i, n, op);
  }
}

// File: aten/src/ATen/native/cpu/Loops.h (lines 105-148)
// Basic loop implementation for non-vectorized cases
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

  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

// File: aten/src/ATen/native/cpu/Loops.h (lines 68-88)
// Execute operation implementation
template <typename func_t,
    std::enable_if_t<!std::is_void_v<typename function_traits<func_t>::result_type>>* = nullptr>
inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    *out_ptr = c10::guts::apply(op, dereference<traits>(
        &data[1],
        &strides[1],
        i));
  }
}

// File: aten/src/ATen/native/TensorCompare.h (lines 36-41)
// Dispatch stub declaration
DECLARE_DISPATCH(
    void (*)(TensorIteratorBase&, c10::Scalar),
    clamp_min_scalar_stub)

// Required headers and dependencies
#define TORCH_ASSERT_NO_OPERATORS
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <ATen/native/Activation.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Dispatch.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/util/irange.h>
#include <cmath>
#include <functional>

namespace at::native {

// Dispatch implementation
DEFINE_DISPATCH(clamp_min_scalar_stub);

} // namespace at::native

/*
 * Complete CPU ReLU Implementation Summary:
 * 
 * 1. relu() function delegates to clamp_min(self, 0)
 * 2. clamp_min() sets up TensorIterator and calls clamp_min_scalar_stub
 * 3. CPU kernel implements scalar operation: std::max(a, min) 
 * 4. Vectorized operation uses vec::clamp_min for SIMD acceleration
 * 5. Core loop infrastructure handles memory access patterns and vectorization
 * 
 * Key optimizations:
 * - SIMD vectorization via Vectorized<T> operations
 * - Efficient memory access patterns in vectorized_loop
 * - Type dispatch for different data types
 * - Contiguous memory optimization
 * - Parallel execution via grain_size parameter
 */