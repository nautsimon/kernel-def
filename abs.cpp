// Complete CPU implementation of PyTorch abs operation
// Extracted from PyTorch source code for self-contained reference

// From: aten/src/ATen/native/Math.h
template <typename T>
inline T abs_impl(T v) {
  return std::abs(v);
}

template <>
[[maybe_unused]] inline uint8_t abs_impl(uint8_t v) {
  return v;
}

// From: aten/src/ATen/native/UnaryOps.h
#include <ATen/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>

using unary_fn = void(*)(at::TensorIteratorBase&);
DECLARE_DISPATCH(unary_fn, abs_stub);

// From: aten/src/ATen/native/UnaryOps.cpp
namespace at::native {

// Helper function for unary operations
template <typename Stub>
static inline Tensor& unary_op_impl_out(Tensor& result, const Tensor& self, Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter.device_type(), iter);
  return result;
}

template <typename Stub, typename ...Args>
static inline Tensor& unary_op_impl_float_out(Tensor& result, const Tensor& self, Stub& stub, Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return result;
}

// Complex to float conversion for abs operation
template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(Tensor& result, const Tensor& self, Stub& stub, bool promotes_integer_to_float) {
    if (self.is_complex() && !result.is_complex()) {
      // Checks if the corresponding float type can be cast to the desired dtype
      const auto float_type = c10::toRealValueType(self.scalar_type());
      TORCH_CHECK(canCast(float_type, result.scalar_type()),
            "result type ", float_type, " can't be cast to the desired output type ",
            result.scalar_type());

      // Runs the function complex->complex, as TensorIterator expects
      Tensor complex_result = at::empty({0}, self.options());
      auto iter = TensorIterator::unary_op(complex_result, self);
      stub(iter.device_type(), iter);

      // Copies the complex result to the actual result and returns it
      at::native::resize_output(result, complex_result.sizes());
      result.copy_(at::real(complex_result));
      return result;
    }

    if (promotes_integer_to_float) {
      return unary_op_impl_float_out(result, self, stub);
    }

    return unary_op_impl_out(result, self, stub);
}

template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(const Tensor& self, OutImpl& out_impl) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

// Main abs function implementations
Tensor& abs_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
}

Tensor abs(const Tensor& self) {
  return unary_op_impl_with_complex_to_float(self, at::abs_out);
}

Tensor& abs_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "In-place abs is not supported for complex tensors.");
  return unary_op_impl_(self, at::abs_out);
}

// Absolute, alias for abs
Tensor& absolute_out(const Tensor& self, Tensor& result) {
  return at::abs_out(result, self);
}

Tensor absolute(const Tensor& self) {
  return self.abs();
}

Tensor& absolute_(Tensor& self) {
  return self.abs_();
}

} // namespace at::native

// From: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
namespace at::native {

inline namespace CPU_CAPABILITY {

#if !defined(C10_MOBILE)
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)                                                 \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND6(                                                 \
            kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)          \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(          \
            kHalf, kBFloat16,                            \
            TYPE, NAME, __VA_ARGS__)
#endif

static void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (dtype == kComplexHalf) {
    using scalar_t = c10::complex<Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t { return abs_impl(opmath_t{a}); });
  } else {
    _AT_DISPATCH_ABS_TYPES(iter.dtype(), "abs_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return abs_impl(a); },
          [=](Vectorized<scalar_t> a) { return a.abs(); });
    });
  }
}

} // inline namespace CPU_CAPABILITY

// Registration
REGISTER_DISPATCH(abs_stub, &CPU_CAPABILITY::abs_kernel)

} // namespace at::native

// From: aten/src/ATen/cpu/vec/vec_base.h
// Vectorized abs implementation snippets for different types
template <typename T>
class Vectorized {
public:
  // Integer types abs implementation
  template <
      typename other_t_abs = T,
      typename std::enable_if_t<
          !is_floating_point_v<other_t_abs> &&
              !c10::is_complex<other_t_abs>::value,
          int> = 0>
  Vectorized<T> abs() const {
    static_assert(std::is_same_v<other_t_abs, T>, "other_t_abs must be T");
    return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
  }
  
  // Floating point types abs implementation
  template <
      typename float_t_abs = T,
      typename std::enable_if_t<is_floating_point_v<float_t_abs>, int> = 0>
  Vectorized<T> abs() const {
    static_assert(std::is_same_v<float_t_abs, T>, "float_t_abs must be T");
    return map([](T x) -> T { return std::abs(x); });
  }
  
  // Complex types abs implementation
  template <
      typename complex_t_abs = T,
      typename std::enable_if_t<c10::is_complex<complex_t_abs>::value, int> = 0>
  Vectorized<T> abs() const {
    static_assert(std::is_same_v<complex_t_abs, T>, "complex_t_abs must be T");
    return map([](T x) { return static_cast<T>(std::abs(x)); });
  }

private:
  Vectorized<T> map(T (*const f)(const T&)) const {
    Vectorized<T> ret;
    for (int64_t i = 0; i != size(); i++) {
      ret[i] = f(values[i]);
    }
    return ret;
  }
};

// From: aten/src/ATen/native/native_functions.yaml dispatch configuration
/*
- func: abs(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs
    SparseCPU, SparseCUDA: abs_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: abs_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs
  tags: [core, pointwise]

- func: abs_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs_
    SparseCPU, SparseCUDA: abs_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: abs_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs_

- func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  dispatch:
    CPU, CUDA, MPS: abs_out
    SparseCPU, SparseCUDA: abs_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMeta: abs_sparse_csr_out
  tags: pointwise
*/