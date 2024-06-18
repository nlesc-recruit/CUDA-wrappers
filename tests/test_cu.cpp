#include <array>
#include <cstring>
#include <iostream>
#include <string>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
// #include <cuda.h>
#include <cudawrappers/cu.hpp>

TEST_CASE("Test cu::Device", "[device]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);

  SECTION("Test Device.getName") {
    const std::string name = device.getName();
    std::cout << "Device name: " << name << std::endl;
    CHECK(name.size() > 0);
  }
}

TEST_CASE("Test context::getDevice", "[device]") {
  cu::init();

  SECTION("Test before initialization") {
    CHECK_THROWS(cu::Context::getCurrent().getDevice());
  }

  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);

  SECTION("Test after initialization") {
    CHECK(device.getName() == cu::Context::getCurrent().getDevice().getName());
  }
}

TEST_CASE("Test copying cu::DeviceMemory and cu::HostMemory using cu::Stream",
          "[memcpy]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);

  SECTION("Test copying a std::array to the device and back") {
    const std::array<int, 3> src = {1, 2, 3};
    std::array<int, 3> tgt = {0, 0, 0};
    const size_t size = sizeof(src);

    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src.data(), size);
    stream.memcpyDtoHAsync(tgt.data(), mem, size);
    stream.synchronize();

    CHECK(src == tgt);
  }

  SECTION("Test copying HostMemory to the device and back") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(!static_cast<bool>(memcmp(src, tgt, size)));
  }
}

TEST_CASE("Test zeroing cu::DeviceMemory", "[zero]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);

  SECTION("Test zeroing cu::DeviceMemory asynchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.zero(mem, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(src, tgt, size)));
  }

  SECTION("Test zeroing cu::DeviceMemory synchronously") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);
    cu::HostMemory src(size);
    cu::HostMemory tgt(size);

    // Populate the memory with values
    int* const src_ptr = static_cast<int*>(src);
    int* const tgt_ptr = static_cast<int*>(tgt);
    for (int i = 0; i < N; i++) {
      src_ptr[i] = i;
      tgt_ptr[i] = 0;
    }
    cu::DeviceMemory mem(size);

    cu::Stream stream;
    stream.memcpyHtoDAsync(mem, src, size);
    stream.synchronize();
    mem.zero(size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(static_cast<bool>(memcmp(src, tgt, size)));
  }

  SECTION("Test cu::RegisteredMemory") {
    const size_t N = 3;
    const size_t size = N * sizeof(int);

    std::vector<int> data_in(size);
    std::vector<int> data_out(size);

    for (size_t i = 0; i < N; i++) {
      data_in[i] = i + 1;
      data_out[i] = 0;
    }

    cu::HostMemory src(data_in.data(), size, 0);
    cu::HostMemory tgt(data_out.data(), size, 0);

    cu::DeviceMemory mem(size);
    cu::Stream stream;

    stream.memcpyHtoDAsync(mem, src, size);
    stream.memcpyDtoHAsync(tgt, mem, size);
    stream.synchronize();

    CHECK(data_in == data_out);
  }

  //  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_DEVICE as host pointer")
  //  {
  //    cu::DeviceMemory mem(sizeof(float), CU_MEMORYTYPE_DEVICE, 0);
  //    float* ptr;
  //    CHECK_THROWS(ptr = mem);
  //  }
  //
  //  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_UNIFIED as host
  //  pointer") {
  //    cu::DeviceMemory mem(sizeof(float), CU_MEMORYTYPE_UNIFIED,
  //                         CU_MEM_ATTACH_GLOBAL);
  //    float* ptr = mem;
  //    CHECK_NOTHROW(ptr[0] = 42.f);
  //  }
  //
  //  SECTION("Test cu::DeviceMemory with invalid CUmemorytype") {
  //    const size_t size = 1024;
  //    CHECK_THROWS(cu::DeviceMemory(size, CU_MEMORYTYPE_ARRAY));
  //    CHECK_THROWS(cu::DeviceMemory(size, CU_MEMORYTYPE_HOST));
  //  }
  //
  //  SECTION("Test cu::DeviceMemory with CU_MEMORYTYPE_DEVICE and flags") {
  //    const size_t size = 1024;
  //    CHECK_NOTHROW(cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, 0));
  //    CHECK_THROWS(
  //        cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, CU_MEM_ATTACH_GLOBAL));
  //    CHECK_THROWS(
  //        cu::DeviceMemory(size, CU_MEMORYTYPE_DEVICE, CU_MEM_ATTACH_HOST));
  //  }
}

TEST_CASE("Test cu::Stream", "[stream]") {
  cu::init();
  cu::Device device(0);
  cu::Context context(hipDeviceScheduleBlockingSync, device);
  cu::Stream stream;

  SECTION("Test memAllocAsync") {
    const size_t size = 1024;
    cu::DeviceMemory mem = stream.memAllocAsync(size);
    CHECK(mem.size() == size);
    CHECK_NOTHROW(stream.memFreeAsync(mem));
    CHECK_NOTHROW(stream.synchronize());
  }
}
