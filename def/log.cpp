// ============================================================================
// Complete PyTorch CPU Log Operation Implementation
// Extracted from PyTorch source code for reference
// ============================================================================

// File: aten/src/ATen/native/native_functions.yaml (log operation definition)
// - func: log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
//   device_check: NoCheck   # TensorIterator
//   structured: True
//   structured_inherits: TensorIteratorBase
//   dispatch:
//     CPU, CUDA, MPS: log_out
//   tags: pointwise

#define TORCH_ASSERT_NO_OPERATORS
#include <cmath>
#include <limits>
#include <type_traits>
#include <atomic>
#include <utility>
#include <variant>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Simplified includes and forward declarations
namespace c10 {
    enum class DeviceType : int8_t { CPU = 0, CUDA = 1, MPS = 4 };
    class Scalar;
    using ScalarType = int;
    const ScalarType kBool = 11;
    const ScalarType kByte = 1;
    const ScalarType kChar = 2;
    const ScalarType kShort = 3;
    const ScalarType kInt = 4;
    const ScalarType kLong = 5;
    const ScalarType kHalf = 6;
    const ScalarType kFloat = 7;
    const ScalarType kDouble = 8;
    const ScalarType kComplexHalf = 9;
    const ScalarType kComplexFloat = 10;
    const ScalarType kComplexDouble = 12;
    const ScalarType kBFloat16 = 13;
}

namespace at {
    class Tensor;
    class TensorBase;
    struct TensorIteratorBase;
    struct TensorIterator;
    
    using ScalarType = c10::ScalarType;
    const auto kBool = c10::kBool;
    const auto kByte = c10::kByte;
    const auto kChar = c10::kChar;
    const auto kShort = c10::kShort;
    const auto kInt = c10::kInt;
    const auto kLong = c10::kLong;
    const auto kHalf = c10::kHalf;
    const auto kFloat = c10::kFloat;
    const auto kDouble = c10::kDouble;
    const auto kComplexHalf = c10::kComplexHalf;
    const auto kComplexFloat = c10::kComplexFloat;
    const auto kComplexDouble = c10::kComplexDouble;
    const auto kBFloat16 = c10::kBFloat16;
    
    bool isComplexType(ScalarType t);
    bool isReducedFloatingType(ScalarType t);
    bool isFloatingType(ScalarType t);
    bool hasMKL();
    void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const std::function<void(int64_t, int64_t)>& f);
    
    template<typename T> struct opmath_type { using type = T; };
    
    // Vectorized type placeholder
    template<typename T>
    struct Vectorized {
        static constexpr int size() { return 8; }
        Vectorized(T val) {}
        Vectorized log() const { return *this; }
    };
}

// ============================================================================
// File: aten/src/ATen/native/DispatchStub.h (dispatch mechanism)
// ============================================================================

namespace at::native {

enum class CPUCapability {
  DEFAULT = 0,
#if defined(HAVE_AVX2_CPU_DEFINITION)
  AVX2 = 1,
  AVX512 = 2,
#endif
  NUM_OPTIONS
};

template <typename FnPtr, typename T>
struct DispatchStub;

struct DispatchStubImpl {
  void* get_call_ptr(c10::DeviceType device_type, void* DEFAULT, void* AVX512 = nullptr, void* AVX2 = nullptr);
  void* choose_cpu_impl(void* DEFAULT, void* AVX512 = nullptr, void* AVX2 = nullptr);
  
  std::atomic<void*> cpu_dispatch_ptr{nullptr};
  void* cuda_dispatch_ptr = nullptr;
  void* hip_dispatch_ptr = nullptr;
  void* mps_dispatch_ptr = nullptr;
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

  DispatchStub() = default;
  DispatchStub(const DispatchStub&) = delete;
  DispatchStub& operator=(const DispatchStub&) = delete;

  template <typename... ArgTypes>
  rT operator()(c10::DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }

private:
  FnPtr get_call_ptr(const c10::DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type, reinterpret_cast<void*>(DEFAULT), 
                       reinterpret_cast<void*>(AVX512), reinterpret_cast<void*>(AVX2))
    );
  }

public:
  static FnPtr DEFAULT;
#ifdef HAVE_AVX512_CPU_DEFINITION
  static FnPtr AVX512;
#endif
#ifdef HAVE_AVX2_CPU_DEFINITION
  static FnPtr AVX2;
#endif

private:
  DispatchStubImpl impl;
};

#define DECLARE_DISPATCH(fn, name)                                                         \
  struct name##_DECLARE_DISPATCH_type : DispatchStub<fn, name##_DECLARE_DISPATCH_type> {   \
    name##_DECLARE_DISPATCH_type() = default;                                              \
    name##_DECLARE_DISPATCH_type(const name##_DECLARE_DISPATCH_type&) = delete;            \
    name##_DECLARE_DISPATCH_type& operator=(const name##_DECLARE_DISPATCH_type&) = delete; \
    ~name##_DECLARE_DISPATCH_type() = default;                                             \
  };                                                                                       \
  extern struct name##_DECLARE_DISPATCH_type name;

#define DEFINE_DISPATCH(name) struct name##_DECLARE_DISPATCH_type name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
  template <> name##_DECLARE_DISPATCH_type::FnPtr DispatchStub<name##_DECLARE_DISPATCH_type::FnPtr, struct name##_DECLARE_DISPATCH_type>::arch = fn;

#ifdef HAVE_AVX512_CPU_DEFINITION
#define REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX512, fn)
#else
#define REGISTER_AVX512_DISPATCH(name, fn)
#endif

#ifdef HAVE_AVX2_CPU_DEFINITION
#define REGISTER_AVX2_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, AVX2, fn)
#else
#define REGISTER_AVX2_DISPATCH(name, fn)
#endif

#ifdef CPU_CAPABILITY_AVX512
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, nullptr)
#else
#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)
#endif

#define ALSO_REGISTER_AVX512_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, CPU_CAPABILITY, fn)

// ============================================================================
// File: aten/src/ATen/native/UnaryOps.h (unary operation declarations)
// ============================================================================

using unary_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(unary_fn, log_stub)

// ============================================================================
// File: aten/src/ATen/native/UnaryOps.cpp (log_out implementation)
// ============================================================================

#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// This expands to:
// TORCH_IMPL_FUNC(log_out) (const Tensor& self, const Tensor& result) {
//   log_stub(device_type(), *this);
// }
CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)

DEFINE_DISPATCH(log_stub);

// ============================================================================
// File: aten/src/ATen/cpu/vml.h (Intel MKL vectorized math library interface)
// ============================================================================

#if AT_MKL_ENABLED() && !defined(__APPLE__)
#include <mkl.h>
#endif

namespace at::vml {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define IMPLEMENT_VML(op)                                               \
  template <typename scalar_t>                                          \
  inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {  \
    using vec_t = Vectorized<vec_scalar_t<scalar_t>>;                   \
    vec::map([](vec_t x) { return x.op(); }, out, in, size);            \
  }

IMPLEMENT_VML(log)

#if AT_MKL_ENABLED() && !defined(__APPLE__)

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

IMPLEMENT_VML_MKL(log, Ln)

#endif

} // namespace
} // namespace at::vml

// ============================================================================
// File: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp (CPU log kernel)
// ============================================================================

namespace at::native {

inline namespace CPU_CAPABILITY {

#if AT_MKL_ENABLED()

template <typename T>
void VmlLog(int64_t N, const T* X, T* Y) {
  constexpr int64_t K = Vectorized<T>::size();
  at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
    using VT = at::opmath_type<T>;
    vec::map(
        [](Vectorized<VT> x_vec) { return x_vec.log(); },
        Y + begin,
        X + begin,
        end - begin);
  });
}

template <>
void VmlLog<float>(int64_t N, const float* X, float* Y) {
  vsLn(N, X, Y);
}

template <>
void VmlLog<double>(int64_t N, const double* X, double* Y) {
  vdLn(N, X, Y);
}

#endif // AT_MKL_ENABLED

// CPU Kernel loop macros and helper functions
template<typename func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, func_t&& vop) {
  // Simplified implementation - applies op element-wise
  // In real PyTorch this uses vectorized operations
}

template<typename func_t>
void cpu_kernel(TensorIteratorBase& iter, func_t&& op) {
  // Simplified implementation - applies op element-wise
}

// Vectorized log implementation using VML lambda approach
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

// Main CPU log kernel implementation using complex kernel macro
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

// This creates the actual log_kernel function
STATIC_IMPLEMENT_COMPLEX_KERNEL(log)

} // CPU_CAPABILITY namespace

// Register the log kernel for different CPU capabilities
STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log)

// This expands to:
// REGISTER_ARCH_DISPATCH(log_stub, CPU_CAPABILITY, &CPU_CAPABILITY::log_kernel)
// ALSO_REGISTER_AVX512_DISPATCH(log_stub, &CPU_CAPABILITY::log_kernel)

} // namespace at::native

// ============================================================================
// Summary of Complete Log CPU Implementation Flow:
// ============================================================================

/*
1. YAML Definition (native_functions.yaml):
   - func: log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
   - dispatch: CPU, CUDA, MPS: log_out

2. High-level Function (UnaryOps.cpp):
   CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)
   
   Expands to:
   TORCH_IMPL_FUNC(log_out)(const Tensor& self, const Tensor& result) {
     log_stub(device_type(), *this);
   }

3. Dispatch Stub (UnaryOps.cpp):
   DEFINE_DISPATCH(log_stub);

4. CPU Kernel Registration (UnaryOpsKernel.cpp):
   STATIC_IMPLEMENT_COMPLEX_KERNEL_WITH_AVX512(log)
   
   This creates log_kernel(TensorIteratorBase& iter) and registers it

5. Core Implementation Logic:
   - Uses VML (Vector Math Library) when available with Intel MKL
   - Falls back to vectorized C++ implementation using AT vectorization
   - Supports both real and complex data types
   - Handles different CPU instruction sets (DEFAULT, AVX2, AVX512)

6. VML Integration:
   - For float: calls MKL's vsLn (single precision natural log)
   - For double: calls MKL's vdLn (double precision natural log)
   - For other types: uses vectorized C++ std::log

Key Features:
- Type dispatch for float, double, half, bfloat16, complex types
- Vectorization for performance
- CPU instruction set optimization (AVX2, AVX512)
- Intel MKL integration when available
- Efficient memory handling for contiguous vs strided tensors
- Parallel processing for large tensors
*/ 