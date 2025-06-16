// =========================================================================
// Complete CPU Implementation of PyTorch Sigmoid Operation
// =========================================================================
// This file contains the complete CPU implementation of the sigmoid operation
// extracted from the PyTorch codebase, including:
// - Main kernel functions executed for sigmoid on CPU
// - Helper functions and macros it depends on
// - Dispatch stubs and registration logic
// - Quantized CPU implementation
// - MKLDNN implementation

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
// Lines 35-66
// =========================================================================

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

inline namespace CPU_CAPABILITY {

using namespace vec;

// =========================================================================
// MAIN SIGMOID CPU KERNEL IMPLEMENTATION
// =========================================================================

static void sigmoid_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "sigmoid_cpu_reduced_float", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            float a0 = static_cast<float>(a);
            return static_cast<float>(1) / (static_cast<float>(1) + std::exp((-a0)));
          },
          [=](Vectorized<scalar_t> a) {
            auto [a0, a1] = convert_to_float<scalar_t>(a);
            a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
            a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
            return convert_from_float<scalar_t>(a0, a1);
          });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(dtype, "sigmoid_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = (Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a.neg().exp()).reciprocal();
            return a;
          });
    });
  }
}

} // namespace CPU_CAPABILITY

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
// Lines 837-882 (Sigmoid Backward)
// =========================================================================

namespace at::native {

inline namespace CPU_CAPABILITY {

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "sigmoid_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b) -> scalar_t {
            return a * std::conj((scalar_t(1) - b) * b);
          },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a * ((one_vec - b) * b).conj();
          });
    });
  } else if (iter.dtype() == kBFloat16) {
    auto one_vec = Vectorized<float>((float)(1));
    cpu_kernel_vec(
        iter,
        [=](BFloat16 a, BFloat16 b) -> BFloat16 {
          float a0 = static_cast<float>(a);
          float b0 = static_cast<float>(b);
          return a0 * (float(1) - b0) * b0;
        },
        [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          auto [a0, a1] = convert_bfloat16_float(a);
          auto [b0, b1] = convert_bfloat16_float(b);
          a0 = a0 * (one_vec - b0) * b0;
          a1 = a1 * (one_vec - b1) * b1;
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        kHalf, iter.dtype(), "sigmoid_backward_cpu", [&]() {
          auto one_vec = Vectorized<scalar_t>((scalar_t)(1));
          cpu_kernel_vec(
              iter,
              [=](scalar_t a, scalar_t b) -> scalar_t {
                return a * (scalar_t(1) - b) * b;
              },
              [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a * (one_vec - b) * b;
              });
        });
  }
}

} // namespace CPU_CAPABILITY

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/UnaryOps.cpp
// Lines 300-347 (Structured Kernel Implementation)
// =========================================================================

namespace at::native {

#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

CREATE_UNARY_TORCH_IMPL_FUNC(sigmoid_out, sigmoid_stub)

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/UnaryOps.cpp
// Lines 796-801 (Special Expit - Sigmoid Alias)
// =========================================================================

namespace at::native {

// special_expit, alias for sigmoid
Tensor& special_expit_out(const Tensor& self, Tensor& result) {
  return at::sigmoid_out(result, self);
}
Tensor special_expit(const Tensor& self) {
  return self.sigmoid();
}

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/UnaryOps.h
// Lines 60-70 (Dispatch Stub Declaration)
// =========================================================================

namespace at::native {

using unary_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(unary_fn, sigmoid_stub);

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/UnaryOps.cpp
// Lines 1036 (Dispatch Stub Definition)
// =========================================================================

namespace at::native {

DEFINE_DISPATCH(sigmoid_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
// Line 845 (Dispatch Registration)
// =========================================================================

namespace at::native {

ALSO_REGISTER_AVX512_DISPATCH(sigmoid_stub, &CPU_CAPABILITY::sigmoid_kernel)

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/mkldnn/UnaryOps.cpp
// Complete MKLDNN Implementation
// =========================================================================

#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sigmoid_native.h>
#endif

#if !AT_MKLDNN_ENABLED()

namespace at::native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  TORCH_CHECK(false, "mkldnn_sigmoid_: ATen not compiled with MKLDNN support");
}

} // namespace at::native

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native {

Tensor mkldnn_sigmoid(const Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return self;
}

} // namespace at::native

#endif // AT_MKLDNN_ENABLED

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/quantized/cpu/qsigmoid.cpp
// Complete Quantized CPU Implementation
// =========================================================================

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <torch/library.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <c10/util/irange.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/sigmoid_native.h>
#endif

#include <algorithm>
#include <utility>

namespace at::native {

DEFINE_DISPATCH(qsigmoid_stub);

#ifdef USE_PYTORCH_QNNPACK
static Tensor qnnpack_sigmoid(
    Tensor input, double output_scale, int64_t output_zero_point) {
  TORCH_CHECK(input.ndimension() > 0, "qnnpack_sigmoid(): Got empty input tensor");
  TORCH_CHECK(input.scalar_type() == c10::kQUInt8,
               "qnnpack_sigmoid(): Expected input data type ",
               toString(c10::kQUInt8),
               " but got ",
               toString(input.scalar_type()));

  Tensor qy;
  initQNNPACK();

  Tensor input_contig = input.contiguous(input.suggest_memory_format());
  size_t num_elems = 1;
  for (const auto i : c10::irange(1, input_contig.ndimension())) {
    num_elems *= input_contig.size(i);
  }

  const auto zero_point = input_contig.q_zero_point();
  const auto scale = input_contig.q_scale();

  pytorch_qnnp_operator_t sigmoid_op{nullptr};
  const pytorch_qnnp_status createStatus = pytorch_qnnp_create_sigmoid_nc_q8(
    num_elems /* channels */,
    zero_point /* input zero point */,
    scale /* input scale */,
    output_zero_point /* output zero point */,
    // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    output_scale /* output scale */,
    std::numeric_limits<uint8_t>::min() /* output min */,
    std::numeric_limits<uint8_t>::max() /* output max */,
    0 /* flags */,
    &sigmoid_op);

  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(sigmoid_op);

  TORCH_INTERNAL_ASSERT(createStatus == pytorch_qnnp_status_success,
                        "failed to create QNNPACK sigmoid operator");
  qy = at::_empty_affine_quantized(
    input_contig.sizes(),
    at::device(kCPU).dtype(input_contig.dtype()),
    output_scale,
    output_zero_point,
    input_contig.suggest_memory_format());

  const pytorch_qnnp_status setupStatus = pytorch_qnnp_setup_sigmoid_nc_q8(
    sigmoid_op,
    input_contig.size(0) /* batch size */,
    (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
    num_elems /* input stride */,
    (uint8_t*)qy.data_ptr<c10::quint8>() /* output data */,
    num_elems /* output stride */);
  TORCH_INTERNAL_ASSERT(setupStatus == pytorch_qnnp_status_success,
                        "failed to setup QNNPACK sigmoid operator");

  pthreadpool_t threadpool = caffe2::pthreadpool_();

  const pytorch_qnnp_status runStatus =
    pytorch_qnnp_run_operator(sigmoid_op, threadpool);

  TORCH_INTERNAL_ASSERT(
    runStatus == pytorch_qnnp_status_success,
    "failed to run QNNPACK sigmoid operator");
  return qy;
}

#endif  // USE_PYTORCH_QNNPACK

// This ALWAYS outputs scale=1.0/256, dtype=quint8
// The zero_point is 0 for qint32 and quint8, but -128 for qint8.
Tensor sigmoid_quantized_cpu(const Tensor& qx) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    constexpr double output_scale = 1.0f / 256.0f;
    constexpr int64_t output_zero_point = 0;
    return qnnpack_sigmoid(qx, output_scale, output_zero_point);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    // Naive implementation: uses dequantize/execute/quantize routine
    // - Output scale is set to 1.0 / 2^(BIT_NUM)
    // - For signed types output zero point is set to 0
    // - For unsigned types output zero point is set to (qmax + qmin) / 2.0
    // See https://stackoverflow.com/a/34448562/3606192 for potential
    // optimizations
    double output_scale = 0.00390625;  // 1.0 / 2^8
    int64_t output_zero_point = 0;
    if (SCALAR_TYPE == at::kQInt32) {
      output_scale = 2.3283064365386963e-10;  // 1.0 / 2^32
    } else if (SCALAR_TYPE == at::kQInt8) {
      output_zero_point = -128;
    }
    qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
  });
  return qy;
}

namespace {

class QSigmoid final {
 public:
  static Tensor run(Tensor qx, double output_scale, int64_t output_zero_point) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    return qnnpack_sigmoid(std::move(qx), output_scale, output_zero_point);
  }
#endif  // USE_PYTORCH_QNNPACK
  Tensor qy;
  qsigmoid_stub(qx.device().type(), qx, qy, output_scale, output_zero_point);
  return qy;
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::sigmoid"), TORCH_FN(QSigmoid::run));
}
} // namespace

}  // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
// Lines 828-870 (Quantized Sigmoid Kernel)
// =========================================================================

namespace at::native {

void qsigmoid_kernel(
    const Tensor& qx, Tensor& qy, double output_scale, int64_t output_zero_point ) {
  int64_t zero_point = qx.q_zero_point();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  float scale = qx.q_scale();
  auto scale_vec = Vectorized<float>(scale);
  auto zero_point_vec = Vectorized<float>((float)zero_point);

  AT_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid", [&]() {
    float inv_output_scale = 1.0 / output_scale;

    qy = at::_empty_affine_quantized(
        qx.sizes(),
        at::device(kCPU).dtype(SCALAR_TYPE).memory_format(qx.suggest_memory_format()),
        output_scale,
        output_zero_point,
        std::nullopt);
    auto iter = TensorIterator::unary_op(qy, qx);

    using Vec = Vectorized<scalar_t>;
    cpu_kernel_vec(
        iter,
        [&](scalar_t value_qx) -> scalar_t {
          const auto value_dx =
              at::native::dequantize_val(scale, zero_point, value_qx);
          const auto value_dy = 1.0f / (1.0 + std::exp((-value_dx)));
          return at::native::quantize_val<scalar_t>(
              output_scale, output_zero_point, value_dy);
        },
        [&](Vec value_qx) -> Vec {
          auto value_dx = value_qx.dequantize(scale_vec, zero_point_vec);
          for (auto & value : value_dx) {
            value = value.neg();
            value = value.exp();
            value = Vectorized<float>(1.0f) + value;
            value = value.reciprocal();
          }
          return Vec::quantize(
              value_dx, output_scale, output_zero_point, inv_output_scale);
        });
  });
}

} // namespace at::native

// =========================================================================
// SOURCE FILE: aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
// Line 4704 (Quantized Sigmoid Dispatch Registration)
// =========================================================================

namespace at::native {

REGISTER_DISPATCH(qsigmoid_stub, &qsigmoid_kernel)

} // namespace at::native

// =========================================================================
// END OF COMPLETE SIGMOID CPU IMPLEMENTATION
// ========================================================================= 