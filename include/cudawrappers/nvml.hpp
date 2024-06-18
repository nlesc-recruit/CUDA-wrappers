#if !defined NVML_H
#define NVML_H
// not avalailable on AMD
#if !defined __HIP_PLATFORM_AMD__

#include <nvml.h>

#include <exception>

#include <cudawrappers/cu.hpp>

namespace nvml {
class Error : public std::exception {
 public:
  explicit Error(nvmlReturn_t result) : _result(result) {}

  const char* what() const noexcept { return nvmlErrorString(_result); }

  operator nvmlReturn_t() const { return _result; }

 private:
  nvmlReturn_t _result;
};

inline void checkNvmlCall(nvmlReturn_t result) {
  if (result != NVML_SUCCESS) throw Error(result);
}

class Context {
 public:
  Context() { checkNvmlCall(nvmlInit()); }

  ~Context() { checkNvmlCall(nvmlShutdown()); }
};

class Device {
 public:
  Device(Context& context, int index) {
    checkNvmlCall(nvmlDeviceGetHandleByIndex(index, &device_));
  }

  Device(Context& context, cu::Device& device) {
    const std::string uuid = device.getUuid();
    nvmlDeviceGetHandleByUUID(uuid.c_str(), &device_);
  }

  void getFieldValues(int valuesCount, nvmlFieldValue_t* values) {
    checkNvmlCall(nvmlDeviceGetFieldValues(device_, valuesCount, values));
  }

 private:
  nvmlDevice_t device_;
};
}  // namespace nvml

#endif  //  __HIP_PLATFORM_AMD__
#endif  // NVML_H