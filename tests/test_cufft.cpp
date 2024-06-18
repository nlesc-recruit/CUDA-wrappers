#include <fstream>
#include <iostream>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cudawrappers/cufft.hpp>

#define FP16_EPSILON 1e-3f
#define FP32_EPSILON 1e-6f

template <typename T>
void generateSignal(T *in, size_t size, size_t patchSize, T signal) {
  for (size_t i = 0; i < patchSize; i++) {
    in[i] = signal;
  }
}

template <typename T>
void generateSignal(T *in, size_t height, size_t width, size_t patchSize,
                    T signal) {
  for (size_t i = 0; i < patchSize; i++) {
    for (size_t j = 0; j < patchSize; j++) {
      in[(width * i) + j] = signal;
    }
  }
}

template <typename T>
void scaleSignal(T *in, T *out, size_t n, float scale) {
  for (size_t i = 0; i < n; i++) {
    out[i].x = static_cast<float>(in[i].x) / scale;
    out[i].y = static_cast<float>(in[i].y) / scale;
  }
}

void compare(float a, float b, double epsilon = FP32_EPSILON) {
  REQUIRE_THAT(a, Catch::Matchers::WithinAbs(b, epsilon));
}

void compare(half a, half b, double epsilon = FP16_EPSILON) {
  compare(__half2float(a), __half2float(b), epsilon);
}

template <typename T>
void compare(T a, T b) {
  compare(a.x, b.x);
  compare(a.y, b.y);
}

template <typename T>
void compare(T *a, T *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    compare(a[i], b[i]);
  }
}

TEST_CASE("Test 1D FFT", "[FFT1D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t size = 256;
  const size_t patchSize = 10;

  SECTION("FP32") {
    const size_t arraySize = size * sizeof(hipfftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<hipfftComplex *>(h_in), size, patchSize, {1, 1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT1D<HIP_C_32F> fft{size};
    fft.setStream(stream);

    fft.execute(d_in, d_out, HIPFFT_FORWARD);
    fft.execute(d_out, d_out2, HIPFFT_BACKWARD);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    hipFloatComplex *in_ptr = static_cast<hipFloatComplex *>(h_in);
    hipFloatComplex *out_ptr = static_cast<hipFloatComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, size, float(size));
    compare(out_ptr, in_ptr, size);
  }

  SECTION("FP16") {
    const size_t arraySize = size * sizeof(half2);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<half2 *>(h_in), 1, size, patchSize, {0.1, 0.1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT1D<HIP_C_16F> fft{size};
    fft.setStream(stream);

    fft.execute(d_in, d_out, HIPFFT_FORWARD);
    fft.execute(d_out, d_out2, HIPFFT_BACKWARD);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    half2 *in_ptr = static_cast<half2 *>(h_in);
    half2 *out_ptr = static_cast<half2 *>(h_out);
    scaleSignal(out_ptr, out_ptr, size, float(size));
    compare(out_ptr, in_ptr, size);
  }
}

TEST_CASE("Test 2D FFT", "[FFT2D]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(CU_CTX_SCHED_BLOCKING_SYNC, device);
  cu::Stream stream;

  const size_t height = 256;
  const size_t width = height;
  const size_t patchSize = 10;

  SECTION("FP32") {
    const size_t arraySize = height * width * sizeof(hipfftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<hipfftComplex *>(h_in), height, width, patchSize,
                   {1, 1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<HIP_C_32F> fft{height, width};
    fft.setStream(stream);

    fft.execute(d_in, d_out, HIPFFT_FORWARD);
    fft.execute(d_out, d_out2, HIPFFT_BACKWARD);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    hipFloatComplex *in_ptr = static_cast<hipFloatComplex *>(h_in);
    hipFloatComplex *out_ptr = static_cast<hipFloatComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }

  SECTION("FP32 batched") {
    const size_t batch = 2;
    const size_t arraySize = batch * height * width * sizeof(hipfftComplex);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    const size_t stride = 1;
    const size_t dist = height * width;

    generateSignal(static_cast<hipFloatComplex *>(h_in), height, width,
                   patchSize, {1, 1});
    generateSignal(static_cast<hipFloatComplex *>(h_in) + dist, height, width,
                   patchSize, {2, 2});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<HIP_C_32F> fft{height, width, stride, dist, batch};
    fft.setStream(stream);

    fft.execute(d_in, d_out, HIPFFT_FORWARD);
    fft.execute(d_out, d_out2, HIPFFT_BACKWARD);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    hipFloatComplex *in_ptr = static_cast<hipFloatComplex *>(h_in);
    hipFloatComplex *out_ptr = static_cast<hipFloatComplex *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }

  SECTION("FP16") {
    const size_t arraySize = height * width * sizeof(__half2);

    cu::HostMemory h_in(arraySize);
    cu::HostMemory h_out(arraySize);
    cu::DeviceMemory d_in(arraySize);
    cu::DeviceMemory d_out(arraySize);
    cu::DeviceMemory d_out2(arraySize);

    generateSignal(static_cast<half2 *>(h_in), height, width, patchSize,
                   {0.1, 0.1});
    stream.memcpyHtoDAsync(d_in, h_in, arraySize);

    cufft::FFT2D<HIP_C_16F> fft{height, width};
    fft.setStream(stream);

    fft.execute(d_in, d_out, HIPFFT_FORWARD);
    fft.execute(d_out, d_out2, HIPFFT_BACKWARD);
    stream.memcpyDtoHAsync(h_out, d_out2, arraySize);
    stream.synchronize();

    half2 *in_ptr = static_cast<half2 *>(h_in);
    half2 *out_ptr = static_cast<half2 *>(h_out);
    scaleSignal(out_ptr, out_ptr, height * width, float(height * width));
    compare(out_ptr, in_ptr, height * width);
  }
}
