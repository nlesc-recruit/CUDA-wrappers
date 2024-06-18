#if !defined NVRTC_H
#define NVRTC_H
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <link.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvrtc {
class Error : public std::exception {
 public:
  explicit Error(hiprtcResult result) : _result(result) {}

  const char *what() const noexcept { return hiprtcGetErrorString(_result); }

  operator hiprtcResult() const { return _result; }

 private:
  hiprtcResult _result;
};

inline void checkNvrtcCall(hiprtcResult result) {
  if (result != HIPRTC_SUCCESS) throw Error(result);
}

inline std::string findIncludePath() {
  std::string path;

  if (dl_iterate_phdr(
          [](struct dl_phdr_info *info, size_t, void *arg) -> int {
            std::string &path = *static_cast<std::string *>(arg);
            path = info->dlpi_name;
            // HIPRTC symbols are also in libamdhip64.so, although they will be removed from there
            // see https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.0/how-to/hip_rtc.html#deprecation-notice
            // check both libraries for now, as linking with hiprtc is not yet required
            return (path.find("libhiprtc.so") != std::string::npos) | (path.find("libamdhip64.so") != std::string::npos);
          },
          &path))
    for (size_t pos; (pos = path.find_last_of("/")) != std::string::npos;) {
      path.erase(pos);  // remove last part of path

      struct stat buffer;
      const std::string filename = path + "/include/hip/hip_runtime.h";
      if (stat(filename.c_str(), &buffer) == 0) {
        return path + "/include";
      }
    }

  throw std::runtime_error("Could not find NVRTC include path");
}

class Program {
 public:
  Program(const std::string &src, const std::string &name,
          const std::vector<std::string> &headers = std::vector<std::string>(),
          const std::vector<std::string> &includeNames =
              std::vector<std::string>()) {
    std::vector<const char *> c_headers;
    std::transform(headers.begin(), headers.end(),
                   std::back_inserter(c_headers),
                   [](const std::string &header) { return header.c_str(); });

    std::vector<const char *> c_includeNames;
    std::transform(
        includeNames.begin(), includeNames.end(),
        std::back_inserter(c_includeNames),
        [](const std::string &includeName) { return includeName.c_str(); });

    checkNvrtcCall(hiprtcCreateProgram(
        &program, src.c_str(), name.c_str(), static_cast<int>(c_headers.size()),
        c_headers.data(), c_includeNames.data()));
  }

  explicit Program(const std::string &filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
      throw std::runtime_error("Error opening file '" + filename +
                               "' in cudawrappers::nvrtc");
    }
    std::string source(std::istreambuf_iterator<char>{ifs}, {});
    checkNvrtcCall(hiprtcCreateProgram(&program, source.c_str(),
                                       filename.c_str(), 0, nullptr, nullptr));
  }

  ~Program() { checkNvrtcCall(hiprtcDestroyProgram(&program)); }

  void compile(const std::vector<std::string> &options) {
    std::vector<const char *> c_options;
    std::transform(options.begin(), options.end(),
                   std::back_inserter(c_options),
                   [](const std::string &option) { return option.c_str(); });
    checkNvrtcCall(hiprtcCompileProgram(
        program, static_cast<int>(c_options.size()), c_options.data()));
  }

  std::vector<char> getPTX() {
    size_t size{};

    checkNvrtcCall(hiprtcGetCodeSize(program, &size));
    std::vector<char> ptx(size);
    checkNvrtcCall(hiprtcGetCode(program, ptx.data()));
    return ptx;
  }

#if CUDA_VERSION >= 11020
  std::vector<char> getCUBIN() {
    size_t size{};
    std::vector<char> cubin;

    checkNvrtcCall(hiprtcGetBitcodeSize(program, &size));
    cubin.resize(size);
    checkNvrtcCall(hiprtcGetBitcode(program, &cubin[0]));
    return cubin;
  }
#endif

  std::string getLog() {
    size_t size{};
    std::string log;

    checkNvrtcCall(hiprtcGetProgramLogSize(program, &size));
    log.resize(size);
    checkNvrtcCall(hiprtcGetProgramLog(program, &log[0]));
    return log;
  }

 private:
  hiprtcProgram program{};
};
}  // namespace nvrtc

#endif
