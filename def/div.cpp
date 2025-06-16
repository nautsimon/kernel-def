// PyTorch CPU Division Operation Implementation
// Extracted from aten/src/ATen/native/cpu/BinaryOpsKernel.cpp and related files
// This file contains the complete CPU implementation for div operations

#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/OpMathType.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/copysign.h>
#include <c10/util/generic_math.h>
#include <c10/core/ScalarType.h>

#include <cmath>
#include <limits>

namespace at::native {

using namespace at::vec;

// From c10/util/generic_math.h - div_floor_floating helper
template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
    __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
      floordiv += scalar_t(1.0);
    }
  } else {
    floordiv = c10::copysign(scalar_t(0), a / b);
  }
  return floordiv;
}

// From c10/util/generic_math.h - div_floor_integer helper
template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_integer(scalar_t a, scalar_t b) {
  if (c10::signs_differ(a, b)) {
    // Subtracts one from the results of truncation division if the
    // divisor and dividend have different sign(bit)s and the remainder of
    // the division is nonzero
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp - binary_op_scalar helper
template <typename scalar_t, typename opmath_t, typename Op>
inline Vectorized<scalar_t> binary_op_scalar(
    const Vectorized<scalar_t>& a,
    opmath_t b,
    const Op& op) {
  Vectorized<opmath_t> vec_a;
  std::tie(vec_a) = convert_to_float<scalar_t>(a);
  Vectorized<opmath_t> vec_b(b);
  return convert_from_float<scalar_t>(op(vec_a, vec_b));
}

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void div_true_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "div_cpu_reduced_float", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      opmath_t b = iter.original_scalar_value<opmath_t>(2);
      iter.remove_operand(2);
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
            return static_cast<opmath_t>(a) / b;
          },
          [=](Vectorized<scalar_t> a) {
            return binary_op_scalar(
                a,
                b,
                [](const Vectorized<opmath_t>& x,
                   const Vectorized<opmath_t>& y) { return x / y; });
          });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kBFloat16, kHalf, dtype, "div_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
                    return a / b;
                  },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a / b;
              });
        });
  }
}

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void div_trunc_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    // TODO: if the divisor is a scalar, rewrite as multiplication by a
    // constant.
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        return a / b;
      });
    });
  } else if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(
        dtype, "div_trunc_cpu_reduced_float", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          opmath_t b = iter.original_scalar_value<opmath_t>(2);
          iter.remove_operand(2);
          cpu_kernel_vec(
              iter,
              [=](scalar_t a)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
                    return std::trunc(static_cast<opmath_t>(a) / b);
                  },
              [=](Vectorized<scalar_t> a) {
                return binary_op_scalar(
                    a,
                    b,
                    [](const Vectorized<opmath_t>& x,
                       const Vectorized<opmath_t>& y) {
                      return (x / y).trunc();
                    });
              });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, dtype, "div_trunc_cpu", [&]() {
          cpu_kernel_vec(
              iter,
              [](scalar_t a, scalar_t b)
                  __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
                    return std::trunc(a / b);
                  },
              [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return (a / b).trunc();
              });
        });
  }
}

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
template <typename scalar_t>
inline Vectorized<scalar_t> div_floor_floating_vec(
    const Vectorized<scalar_t>& a,
    const Vectorized<scalar_t>& b) {
  using vec_t = Vectorized<scalar_t>;
  const auto basic_div = a / b;
  vec_t inf(std::numeric_limits<scalar_t>::infinity());
  auto mod = a.fmod(b);
  // Fixup for a case that isn't properly handled by Sleef_fmod
  auto floor = vec_t::blendv(a - mod, a, (basic_div.abs() == inf) & (a.abs() != inf));
  auto div = floor / b;
  const auto zero = vec_t(0);
  auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
  const auto one = vec_t(1);
  div = vec_t::blendv(div, div - one, mask);
  auto floordiv = div.floor();
  mask = (div - floordiv) > vec_t(0.5);
  floordiv = vec_t::blendv(floordiv, floordiv + one, mask);
  floordiv = vec_t::blendv(floordiv, zero.copysign(basic_div), div == zero);
  floordiv = vec_t::blendv(floordiv, basic_div, b == zero);
  return floordiv;
}

#if defined(CPU_CAPABILITY_SVE256) && defined(__ARM_FEATURE_BF16)

// Since sve lacks sufficient bf16 intrinsics, do the calculations in f32 to
// avoid rounding errors. This should not cause performance issues as
// most of the used instructions would be cast to f32 vectors anyway.
template<>
inline Vectorized<c10::BFloat16> div_floor_floating_vec(
  const Vectorized<c10::BFloat16>& a,
  const Vectorized<c10::BFloat16>& b) {
  auto [a1, a2] = convert_bfloat16_float(a);
  auto [b1, b2] = convert_bfloat16_float(b);

  auto res1 = div_floor_floating_vec(a1, b1);
  auto res2 = div_floor_floating_vec(a2, b2);

  return convert_float_bfloat16(res1, res2);
}

#endif

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void div_floor_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // In the special case of unsigned integer division, floor division is
    // equivalent to truncation division (since the signs of the divisor and
    // dividend are always the same)
    return div_trunc_kernel(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        return c10::div_floor_integer(a, b);
      });
    });
  } else {
    // See NOTE: [Floor Division in Python]
    if (iter.is_scalar(2) && iter.data_ptr(2) != nullptr && at::isReducedFloatingType(dtype)) {
      AT_DISPATCH_REDUCED_FLOATING_TYPES(
          dtype, "div_floor_cpu_reduced_float", [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            opmath_t b = iter.original_scalar_value<opmath_t>(2);
            iter.remove_operand(2);
            using vec_t = Vectorized<opmath_t>;
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t {
                  return c10::div_floor_floating(static_cast<opmath_t>(a), b);
                },
                [=](Vectorized<scalar_t> a) {
                  return binary_op_scalar(
                      a, b, [](const vec_t& x, const vec_t& y) {
                        return div_floor_floating_vec(x, y);
                      });
                });
          });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kBFloat16, kHalf, dtype, "div_floor_cpu", [&]() {
            using vec_t = Vectorized<scalar_t>;
            cpu_kernel_vec(
                iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                  return c10::div_floor_floating(a, b);
                },
                [](vec_t a, vec_t b) -> vec_t {
                  return div_floor_floating_vec(a, b);
                });
          });
    }
  }
}

// From aten/src/ATen/native/BinaryOps.cpp - TORCH_IMPL_FUNC implementations
namespace {

// This is the dispatch stub definition - from aten/src/ATen/native/BinaryOps.h
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(div_trunc_stub);
DEFINE_DISPATCH(div_floor_stub);

} // namespace

// From aten/src/ATen/native/BinaryOps.cpp
TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  div_true_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out_mode) (
  const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    div_floor_stub(device_type(), *this);
  }
}

// From aten/src/ATen/native/cpu/BinaryOpsKernel.cpp - dispatch registrations
REGISTER_DISPATCH(div_true_stub, &div_true_kernel)
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel)
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel)

} // namespace at::native 