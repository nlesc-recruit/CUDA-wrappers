#if !defined CUFFT_H
#define CUFFT_H

#include <hip/hip_fp16.h>
#include <hipfft/hipfft.h>
#include <hipfft/hipfftXt.h>

#include <exception>

#include "cudawrappers/cu.hpp"

/*
 * Error handling helper function, copied from cuda-samples Common/helper_cuda.h
 */
static const char *_cudaGetErrorEnum(hipfftResult error) {
  switch (error) {
    case HIPFFT_SUCCESS:
      return "HIPFFT_SUCCESS";

    case HIPFFT_INVALID_PLAN:
      return "HIPFFT_INVALID_PLAN";

    case HIPFFT_ALLOC_FAILED:
      return "HIPFFT_ALLOC_FAILED";

    case HIPFFT_INVALID_TYPE:
      return "HIPFFT_INVALID_TYPE";

    case HIPFFT_INVALID_VALUE:
      return "HIPFFT_INVALID_VALUE";

    case HIPFFT_INTERNAL_ERROR:
      return "HIPFFT_INTERNAL_ERROR";

    case HIPFFT_EXEC_FAILED:
      return "HIPFFT_EXEC_FAILED";

    case HIPFFT_SETUP_FAILED:
      return "HIPFFT_SETUP_FAILED";

    case HIPFFT_INVALID_SIZE:
      return "HIPFFT_INVALID_SIZE";

    case HIPFFT_UNALIGNED_DATA:
      return "HIPFFT_UNALIGNED_DATA";

    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      return "HIPFFT_INCOMPLETE_PARAMETER_LIST";

    case HIPFFT_INVALID_DEVICE:
      return "HIPFFT_INVALID_DEVICE";

    case HIPFFT_PARSE_ERROR:
      return "HIPFFT_PARSE_ERROR";

    case HIPFFT_NO_WORKSPACE:
      return "HIPFFT_NO_WORKSPACE";

    case HIPFFT_NOT_IMPLEMENTED:
      return "HIPFFT_NOT_IMPLEMENTED";

    case HIPFFT_NOT_SUPPORTED:
      return "HIPFFT_NOT_SUPPORTED";
  }

  return "<unknown>";
}

namespace cufft {

/*
 * Error
 */
class Error : public std::exception {
 public:
  explicit Error(hipfftResult result) : result_(result) {}

  const char *what() const noexcept override {
    return _cudaGetErrorEnum(result_);
  }

  operator hipfftResult() const { return result_; }

 private:
  hipfftResult result_;
};

/*
 * FFT
 */
class FFT {
 public:
  FFT() = default;
  FFT &operator=(FFT &) = delete;
  FFT(FFT &) = delete;
  FFT &operator=(FFT &&other) noexcept {
    if (&other != this) {
      plan_ = other.plan_;
      other.plan_ = 0;
    }
    return *this;
  }
  FFT(FFT &&other) noexcept { *this = std::move(other); }

  ~FFT() { checkCuFFTCall(hipfftDestroy(plan_)); }

  void setStream(cu::Stream &stream) {
    checkCuFFTCall(hipfftSetStream(plan_, stream));
  }

  void execute(cu::DeviceMemory &in, cu::DeviceMemory &out, int direction) {
    void *in_ptr = reinterpret_cast<void *>(static_cast<hipDeviceptr_t>(in));
    void *out_ptr = reinterpret_cast<void *>(static_cast<hipDeviceptr_t>(out));
    checkCuFFTCall(hipfftXtExec(plan_, in_ptr, out_ptr, direction));
  }

 protected:
  void checkCuFFTCall(hipfftResult result) {
    if (result != HIPFFT_SUCCESS) {
      throw Error(result);
    }
  }

  hipfftHandle *plan() { return &plan_; }

 private:
  hipfftHandle plan_{};
};

/*
 * FFT1D
 */
template <hipDataType T>
class FFT1D : public FFT {
 public:
  __host__ FFT1D(int nx) = delete;
  __host__ FFT1D(int nx, int batch) = delete;
};

template <>
FFT1D<HIP_C_32F>::FFT1D(int nx, int batch) {
  checkCuFFTCall(hipfftCreate(plan()));
  checkCuFFTCall(hipfftPlan1d(plan(), nx, HIPFFT_C2C, batch));
}

template <>
FFT1D<HIP_C_32F>::FFT1D(int nx) : FFT1D(nx, 1) {}

template <>
FFT1D<HIP_C_16F>::FFT1D(int nx, int batch) {
  checkCuFFTCall(hipfftCreate(plan()));
  const int rank = 1;
  size_t ws = 0;
  std::array<long long, 1> n{nx};
  long long int idist = 1;
  long long int odist = 1;
  int istride = 1;
  int ostride = 1;
  checkCuFFTCall(hipfftXtMakePlanMany(*plan(), rank, n.data(), nullptr, istride,
                                      idist, HIP_C_16F, nullptr, ostride, odist,
                                      HIP_C_16F, batch, &ws, HIP_C_16F));
}

template <>
FFT1D<HIP_C_16F>::FFT1D(int nx) : FFT1D(nx, 1) {}

/*
 * FFT2D
 */
template <hipDataType T>
class FFT2D : public FFT {
 public:
  __host__ FFT2D(int nx, int ny) = delete;
  __host__ FFT2D(int nx, int ny, int stride, int dist, int batch) = delete;
};

template <>
FFT2D<HIP_C_32F>::FFT2D(int nx, int ny) {
  checkCuFFTCall(hipfftCreate(plan()));
  checkCuFFTCall(hipfftPlan2d(plan(), nx, ny, HIPFFT_C2C));
}

template <>
FFT2D<HIP_C_32F>::FFT2D(int nx, int ny, int stride, int dist, int batch) {
  checkCuFFTCall(hipfftCreate(plan()));
  std::array<int, 2> n{nx, ny};
  checkCuFFTCall(hipfftPlanMany(plan(), 2, n.data(), n.data(), stride, dist,
                                n.data(), stride, dist, HIPFFT_C2C, batch));
}

template <>
FFT2D<HIP_C_16F>::FFT2D(int nx, int ny, int stride, int dist, int batch) {
  checkCuFFTCall(hipfftCreate(plan()));
  const int rank = 2;
  size_t ws = 0;
  std::array<long long, 2> n{nx, ny};
  int istride = stride;
  int ostride = stride;
  long long int idist = dist;
  long long int odist = dist;
  checkCuFFTCall(hipfftXtMakePlanMany(*plan(), rank, n.data(), nullptr, istride,
                                      idist, HIP_C_16F, nullptr, ostride, odist,
                                      HIP_C_16F, batch, &ws, HIP_C_16F));
}

template <>
FFT2D<HIP_C_16F>::FFT2D(int nx, int ny) : FFT2D(nx, ny, 1, nx * ny, 1) {}

}  // namespace cufft

#endif  // CUFFT_H
