#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
// #include <cuda.h>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

bool arrays_equal(const float *a, const float *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (a[i] != Approx(b[i]).epsilon(1e-6)) {
      return false;
    }
  }
  return true;
}

void initialize_arrays(float *a, float *b, float *c, float *r, int N) {
  for (int i = 0; i < N; i++) {
    a[i] = 1.0 + i;
    b[i] = 2.0 - (N - i);
    c[i] = 0.0;
    r[i] = a[i] + b[i];
  }
}

TEST_CASE("Vector add") {
  const std::string kernel = R"(
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        c[i] = a[i] + b[i];
      }
    }
  )";

  cu::init();
  const int N = 1024;
  const size_t bytesize = N * sizeof(float);

  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);

  cu::Stream stream;

  std::vector<std::string> options = {};
  nvrtc::Program program(kernel, "vector_add_kernel.cu");
  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    std::cerr << program.getLog();
    throw;
  }

  cu::Module module(static_cast<const void *>(program.getPTX().data()));
  cu::Function function(module, "vector_add");

  SECTION("Run kernel") {
    cu::HostMemory h_a(bytesize);
    cu::HostMemory h_b(bytesize);
    cu::HostMemory h_c(bytesize);
    std::vector<float> reference_c(N);

    initialize_arrays(static_cast<float *>(h_a), static_cast<float *>(h_b),
                      static_cast<float *>(h_c), reference_c.data(), N);

    cu::DeviceMemory d_a(bytesize);
    cu::DeviceMemory d_b(bytesize);
    cu::DeviceMemory d_c(bytesize);

    stream.memcpyHtoDAsync(d_a, h_a, bytesize);
    stream.memcpyHtoDAsync(d_b, h_b, bytesize);
    std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                            d_b.parameter(), &N};
    stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
    stream.memcpyDtoHAsync(h_c, d_c, bytesize);
    stream.synchronize();

    CHECK(arrays_equal(h_c, reference_c.data(), N));
  }
  /*
    SECTION("Run kernel with managed memory") {
      cu::DeviceMemory d_a(bytesize, CU_MEMORYTYPE_UNIFIED, CU_MEM_ATTACH_HOST);
      cu::DeviceMemory d_b(bytesize, CU_MEMORYTYPE_UNIFIED, CU_MEM_ATTACH_HOST);
      cu::DeviceMemory d_c(bytesize, CU_MEMORYTYPE_UNIFIED, CU_MEM_ATTACH_HOST);

      float *h_a = d_a;
      float *h_b = d_b;
      float *h_c = d_c;
      std::vector<float> reference_c(N);

      initialize_arrays(h_a, h_b, h_c, reference_c.data(), N);

      std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                              d_b.parameter(), &N};
      stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
      stream.synchronize();

      CHECK(arrays_equal(h_c, reference_c.data(), N));
    }

    SECTION("Run kernel with managed memory and prefetch") {
      if (device.getAttribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS)) {
        cu::DeviceMemory d_a(bytesize, CU_MEMORYTYPE_UNIFIED,
                             CU_MEM_ATTACH_GLOBAL);
        cu::DeviceMemory d_b(bytesize, CU_MEMORYTYPE_UNIFIED,
                             CU_MEM_ATTACH_GLOBAL);
        cu::DeviceMemory d_c(bytesize, CU_MEMORYTYPE_UNIFIED,
                             CU_MEM_ATTACH_GLOBAL);

        float *h_a = d_a;
        float *h_b = d_b;
        float *h_c = d_c;
        std::vector<float> reference_c(N);

        initialize_arrays(h_a, h_b, h_c, reference_c.data(), N);

        std::vector<const void *> parameters = {d_c.parameter(),
    d_a.parameter(), d_b.parameter(), &N}; stream.memPrefetchAsync(d_a,
    bytesize, device); stream.memPrefetchAsync(d_b, bytesize, device);
        stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
        stream.memPrefetchAsync(d_c, bytesize);
        stream.synchronize();

        CHECK(arrays_equal(h_c, reference_c.data(), N));
      }
    }
  */

  SECTION("Run kernel with asynchronously allocated memory") {
    cu::HostMemory h_a(bytesize);
    cu::HostMemory h_b(bytesize);
    cu::HostMemory h_c(bytesize);
    std::vector<float> reference_c(N);

    initialize_arrays(static_cast<float *>(h_a), static_cast<float *>(h_b),
                      static_cast<float *>(h_c), reference_c.data(), N);

    const size_t memory_free = context.getFreeMemory();

    cu::DeviceMemory d_a = stream.memAllocAsync(bytesize);
    cu::DeviceMemory d_b = stream.memAllocAsync(bytesize);
    cu::DeviceMemory d_c = stream.memAllocAsync(bytesize);

    stream.memcpyHtoDAsync(d_a, h_a, bytesize);
    stream.memcpyHtoDAsync(d_b, h_b, bytesize);
    std::vector<const void *> parameters = {d_c.parameter(), d_a.parameter(),
                                            d_b.parameter(), &N};
    stream.launchKernel(function, 1, 1, 1, N, 1, 1, 0, parameters);
    stream.memcpyDtoHAsync(h_c, d_c, bytesize);
    stream.memFreeAsync(d_a);
    stream.memFreeAsync(d_b);
    stream.memFreeAsync(d_c);
    stream.synchronize();
    /*
        if (!device.getAttribute(CU_DEVICE_ATTRIBUTE_INTEGRATED)) {
          // The following CHECK is disabled since we cannot make
          // sure that the test is using the GPU exclusively:
          // CHECK(memory_free == context.getFreeMemory());
        }
    */

    CHECK(arrays_equal(h_c, reference_c.data(), N));
  }
}
