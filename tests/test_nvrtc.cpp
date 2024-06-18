#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <cudawrappers/nvrtc.hpp>

TEST_CASE("Test nvrtc::Program", "[program]") {
  const std::string kernel = R"(
    __global__ void vector_add(float *c, float *a, float *b, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) {
        c[i] = a[i] + b[i];
      }
    }
  )";

  nvrtc::Program program(kernel, "kernel.cu");

  SECTION("Test Program.compile") {
    const std::vector<std::string> options = {"--generate-line-info"};
    CHECK_NOTHROW(program.compile(options));
  }

  SECTION("Test Program.getPTX") {
    const std::vector<char> ptx{program.getPTX()};
    CHECK(ptx.size() > 0);
  }
}

extern const char _binary_tests_kernels_vector_add_kernel_cu_start,
    _binary_tests_kernels_vector_add_kernel_cu_end;

TEST_CASE("Test nvrtc::Program embedded source", "[program]") {
  const std::string kernel(&_binary_tests_kernels_vector_add_kernel_cu_start,
                           &_binary_tests_kernels_vector_add_kernel_cu_end);
  nvrtc::Program program(kernel, "vector_add_kernel.cu");

  SECTION("Test Program.compile") {
    const std::vector<std::string> options = {"--generate-line-info"};
    CHECK_NOTHROW(program.compile(options));
  }

  SECTION("Test Program.getPTX") {
    const std::vector<char> ptx{program.getPTX()};
    CHECK(ptx.size() > 0);
  }
}

TEST_CASE("Test nvrtc::findIncludePath", "[helper]") {
  const std::string path = nvrtc::findIncludePath();
  CHECK(path.find("include") != std::string::npos);
}