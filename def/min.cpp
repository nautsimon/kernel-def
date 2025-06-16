// ==============================================================================
// COMPLETE CPU IMPLEMENTATION OF MIN OPERATIONS IN PYTORCH
// ==============================================================================
// This file contains the complete CPU implementation of min operations,
// extracted from the PyTorch codebase. It includes all kernel functions,
// helper functions, and dispatch logic necessary for CPU execution.
//
// Sources:
// - aten/src/ATen/native/cpu/TensorCompareKernel.cpp
// - aten/src/ATen/native/cpu/ReduceOpsKernel.cpp  
// - aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp
// - aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
// - aten/src/ATen/native/SharedReduceOps.h
// - aten/src/ATen/native/cpu/zmath.h
// - aten/src/ATen/native/ReduceOpsUtils.h
// - aten/src/ATen/cpu/vec/vec_base.h
// - aten/src/ATen/native/cpu/Loops.h
// - aten/src/ATen/native/cpu/Reduce.h
// ==============================================================================

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NumericUtils.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#include <c10/util/Optional.h>
#include <c10/core/ScalarType.h>

#include <algorithm>
#include <numeric>
#include <iterator>
#include <cmath>

namespace at::native {

// ==============================================================================
// HELPER FUNCTIONS FROM zmath.h
// ==============================================================================

namespace {

// From aten/src/ATen/native/cpu/zmath.h
template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::min(a, b);
  }
}

template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE min_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) < std::abs(b) ? a : b;
  }
}

template <typename TYPE, std::enable_if_t<!c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a) || _isnan<TYPE>(b)) {
    return std::numeric_limits<TYPE>::quiet_NaN();
  } else {
    return std::max(a, b);
  }
}

template <typename TYPE, std::enable_if_t<c10::is_complex<TYPE>::value, int> = 0>
inline TYPE max_impl (TYPE a, TYPE b) {
  if (_isnan<TYPE>(a)) {
    return a;
  } else if (_isnan<TYPE>(b)) {
    return b;
  } else {
    return std::abs(a) > std::abs(b) ? a : b;
  }
}

} // anonymous namespace

// ==============================================================================
// BOUNDS HELPERS FROM ReduceOpsUtils.h
// ==============================================================================

// From aten/src/ATen/native/ReduceOpsUtils.h
template <typename scalar_t>
constexpr scalar_t upper_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? lim::infinity() : lim::max();
}

template <typename scalar_t>
constexpr scalar_t lower_bound() {
  using lim = std::numeric_limits<scalar_t>;
  return lim::has_infinity ? -lim::infinity() : lim::lowest();
}

// ==============================================================================
// SHARED REDUCE OPS STRUCTURES FROM SharedReduceOps.h
// ==============================================================================

namespace detail {

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T1, typename T2> using pair = thrust::pair<T1, T2>;
#else
template <typename T1, typename T2> using pair = std::pair<T1, T2>;
#endif

template <typename scalar_t>
struct LessOrNan {
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else min(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a < b);
  }
};

template <typename scalar_t>
struct GreaterOrNan {
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else max(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
  }
};

template <typename comp_t>
struct MinMaxReductionOps {
  using scalar_t = typename binary_function_traits<comp_t>::arg1_t;
  using index_t = int64_t;
  using arg_t = detail::pair<scalar_t, index_t>;

  static C10_DEVICE arg_t project(arg_t arg) {
    return arg;
  }

  static C10_DEVICE arg_t reduce(arg_t arg, scalar_t val, int64_t idx) {
    return comp_t{}(arg.first, val, arg.second, idx) ? arg : arg_t(val, idx);
  }

  static C10_DEVICE arg_t combine(arg_t a, arg_t b) {
    return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
  }

  static C10_DEVICE arg_t translate_idx(arg_t a, int64_t base_idx) {
    return {a.first, a.second + base_idx};
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  static C10_DEVICE arg_t warp_shfl_down(arg_t arg, int offset) {
    return arg_t(WARP_SHFL_DOWN(arg.first, offset),
                 WARP_SHFL_DOWN(arg.second, offset));
  }
#endif
};

template <typename comp_t>
struct ArgReductionOps : public MinMaxReductionOps<comp_t> {
  using typename MinMaxReductionOps<comp_t>::scalar_t;
  using typename MinMaxReductionOps<comp_t>::index_t;
  using typename MinMaxReductionOps<comp_t>::arg_t;

  static C10_DEVICE index_t project(arg_t arg) {
    return arg.second;
  }
};

} // namespace detail

template <typename scalar_t>
struct ArgMaxOps :
  public detail::ArgReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template <typename scalar_t>
struct ArgMinOps :
  public detail::ArgReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MinOps :
  public detail::MinMaxReductionOps<detail::LessOrNan<scalar_t>> {
};

template <typename scalar_t>
struct MaxOps :
  public detail::MinMaxReductionOps<detail::GreaterOrNan<scalar_t>> {
};

template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

// ==============================================================================
// VECTORIZED MINIMUM FUNCTIONS FROM vec_base.h
// ==============================================================================

// From aten/src/ATen/cpu/vec/vec_base.h
template <
    class T,
    typename std::enable_if_t<!c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (a[i] < b[i]) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

template <
    class T,
    typename std::enable_if_t<c10::is_complex<T>::value, int> = 0>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
  Vectorized<T> c;
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    c[i] = (std::abs(a[i]) < std::abs(b[i])) ? a[i] : b[i];
    if (_isnan(a[i])) {
      // If either input is NaN, propagate a NaN.
      // NOTE: The case where b[i] was NaN is handled correctly by the naive
      // ternary operator above.
      c[i] = a[i];
    }
  }
  return c;
}

// ==============================================================================
// HELPER FUNCTIONS FOR DIMENSION HANDLING
// ==============================================================================

namespace {

// From aten/src/ATen/native/cpu/TensorCompareKernel.cpp
int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}

int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}

} // anonymous namespace

// ==============================================================================
// MAIN MIN KERNEL IMPLEMENTATIONS
// ==============================================================================

namespace {

// From aten/src/ATen/native/cpu/TensorCompareKernel.cpp
template <typename scalar_t, typename scalar_t_2 = int64_t, typename loop1d_t>
static inline void compare_base_kernel_core(
    const Tensor& result1,
    const Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const loop1d_t& loop) {
  auto result1_strides = result1.strides();
  auto result1_sizes = result1.sizes();
  auto result2_strides = result2.strides();
  auto result2_sizes = result2.sizes();
  auto self_strides = self.strides();
  auto self_sizes = self.sizes();

  auto inner_shape = DimVector(self_sizes);
  auto inner_strides = DimVector(self_strides);
  auto result1_inner_strides = DimVector(result1_strides);
  auto result2_inner_strides = DimVector(result2_strides);
  inner_shape[dim] = 1;
  inner_strides[dim] = 0;
  result1_inner_strides[dim] = 0;
  result2_inner_strides[dim] = 0;

  TensorIterator iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(result1.as_strided(inner_shape, result1_inner_strides))
    .add_output(result2.as_strided(inner_shape, result2_inner_strides))
    .add_input(self.as_strided(inner_shape, inner_strides))
    .build();

  auto loop_2d = [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    auto* result1_data_bytes = data[0];
    auto* result2_data_bytes = data[1];
    const auto* self_data_bytes = data[2];
    auto result1_step = strides[0];
    auto result2_step = strides[1];
    auto self_step = strides[2];
    auto self_dim_stride = ensure_nonempty_stride(self, dim);

    for (const auto i : c10::irange(size1)) {
      loop(
        reinterpret_cast<scalar_t*>(result1_data_bytes),
        reinterpret_cast<scalar_t_2*>(result2_data_bytes),
        reinterpret_cast<const scalar_t*>(self_data_bytes),
        self_dim_stride);
      result1_data_bytes += result1_step;
      result2_data_bytes += result2_step;
      self_data_bytes += self_step;
    }
  };

  iter.for_each(loop_2d);
}

template <typename scalar_t, typename scalar_t_2=int64_t, typename func_t>
static inline void compare_base_kernel(const Tensor& result1, const Tensor& result2,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const func_t& f) {
  auto self_dim_size = ensure_nonempty_size(self, dim);
  
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result1_data_bytes = data[0];
    auto* result2_data_bytes = data[1];
    const auto* self_data_bytes = data[2];
    auto result1_step = strides[0];
    auto result2_step = strides[1];
    auto self_step = strides[2];
    auto self_dim_stride = ensure_nonempty_stride(self, dim);

    for (const auto i : c10::irange(n)) {
      f(reinterpret_cast<scalar_t*>(result1_data_bytes + i * result1_step),
        reinterpret_cast<scalar_t_2*>(result2_data_bytes + i * result2_step),
        reinterpret_cast<const scalar_t*>(self_data_bytes + i * self_step),
        self_dim_stride);
    }
  };

  compare_base_kernel_core<scalar_t, scalar_t_2>(
      result1, result2, self, dim, keepdim, loop);
}

// From aten/src/ATen/native/cpu/TensorCompareKernel.cpp  
static void min_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half, ScalarType::BFloat16, ScalarType::Bool, self.scalar_type(), "min_cpu", [&] {
    compare_base_kernel<scalar_t>(result, indice, self, dim, keepdim, [&] (
      scalar_t* result_data, int64_t* indice_data,
      const scalar_t* self_data, auto self_dim_stride) {
        using value_t = typename c10::scalar_value_type<scalar_t>::type;
        value_t (*zabs_)(scalar_t) = zabs<scalar_t, value_t>;
        scalar_t min_number = c10::load(self_data);
        int64_t index = 0;
        for (const auto i : c10::irange(self_dim_size)) {
          scalar_t value = c10::load(&self_data[i * self_dim_stride]);
          if (!(zabs_(value) >= zabs_(min_number))) {
            min_number = value;
            index = i;
            if (_isnan<scalar_t>(value)) {
              break;
            }
          }
        }
        *result_data = min_number;
        *indice_data = index;
      }
    );
  });
}

// From aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
static void min_values_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == kLong) {
    // This case is special because of Vectorized<int64_t> does not
    // handle upper_bound<int64_t>().
    // See: https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}

// From aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
static void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmin_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMinOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(upper_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), 0));
  });
}

// ==============================================================================
// MIN ALL REDUCE IMPLEMENTATIONS
// ==============================================================================

// For operation not support in avx/avx2
template <typename scalar_t, typename func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op) {
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t ident) -> scalar_t {
      scalar_t partial_out = ident;
      for (const auto i : c10::irange(start, end)) {
         partial_out = op(partial_out, input_data[i]);
      }
      return partial_out;
    }, op);
  output.fill_(result);
}

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop) {
  using Vec = Vectorized<opmath_type<scalar_t>>;
  const int64_t input_numel = input.numel();
  auto input_data = input.const_data_ptr<scalar_t>();
  // NOTE: parallel_reduce not support bool type
  scalar_t result = at::parallel_reduce(0, input_numel, internal::GRAIN_SIZE, ident_v,
    [&](int64_t start, int64_t end, const scalar_t /*ident*/) -> scalar_t {
      scalar_t partial_out = vec::reduce_all<scalar_t>(
        [=](Vec x, Vec y) { return vop(x, y); },
        input_data + start,
        end - start);
      return partial_out;
    }, op);
  output.fill_(result);
}

// From aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp
static void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Bool) {
    TensorIterator iter = TensorIteratorConfig()
      .add_input(input)
      .build();
    bool result_data  = true;
    cpu_serial_kernel(iter, [&](const bool a) -> void {
      result_data = result_data && a;
    });
    result.fill_(result_data);
  } else if(input.scalar_type() == ScalarType::Long) {
    // for int64_t, vectorized implementation have performance issue,
    // just use scalar path
    reduce_all_impl<int64_t>(result, input, upper_bound<int64_t>(),
      [=](int64_t a, int64_t b) -> int64_t { return min_impl(a, b); });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "min_all", [&] {
      using Vec = Vectorized<opmath_type<scalar_t>>;
      reduce_all_impl_vec<scalar_t>(result, input, upper_bound<scalar_t>(),
        [=] (scalar_t a , scalar_t b) -> scalar_t { return min_impl(a, b); },
        [=](Vec a, Vec b) -> Vec { return minimum(a, b); });
    });
  }
}

// ==============================================================================
// BINARY MIN OPERATIONS
// ==============================================================================

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp  
void minimum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [](bool a, bool b) -> bool { return a && b; });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return at::vec::minimum(a, b);
          });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "minimum_cpu",
        [&]() {
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b) -> scalar_t {
                if (a != a || b != b) {
                  return std::numeric_limits<scalar_t>::quiet_NaN();
                } else {
                  return std::min(a, b);
                }
              },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return at::vec::minimum(a, b);
              });
        });
  }
}

void fmin_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "fmin_cpu",
        [&]() {
          cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return std::fmin(a, b);
          });
        });
  } else {
    minimum_kernel(iter);
  }
}

} // anonymous namespace

// ==============================================================================
// DISPATCH STUB REGISTRATIONS
// ==============================================================================

// From aten/src/ATen/native/cpu/TensorCompareKernel.cpp
REGISTER_DISPATCH(min_stub, &min_kernel_impl)

// From aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl)
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl)

// From aten/src/ATen/native/cpu/ReduceAllOpsKernel.cpp
REGISTER_DISPATCH(min_all_stub, &min_all_kernel_impl)

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
REGISTER_DISPATCH(minimum_stub, &minimum_kernel)
REGISTER_DISPATCH(fmin_stub, &fmin_kernel)

} // namespace at::native

// ==============================================================================
// END OF COMPLETE MIN CPU IMPLEMENTATION
// ==============================================================================