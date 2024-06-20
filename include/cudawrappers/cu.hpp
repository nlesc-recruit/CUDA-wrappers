#if !defined CU_WRAPPER_H
#define CU_WRAPPER_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cudawrappers/macros.hpp>

#ifdef __HIP__
typedef uint32_t cuuint32_t;
#endif

namespace cu {
class Error : public std::exception {
 public:
  explicit Error(hipError_t result) : _result(result) {}

  const char *what() const noexcept {
    const char *str{};
    return hipDrvGetErrorString(_result, &str) != hipErrorInvalidValue
               ? str
               : "unknown error";
  }

  operator hipError_t() const { return _result; }

 private:
  hipError_t _result;
};

inline void checkCudaCall(hipError_t result) {
  if (result != hipSuccess) throw Error(result);
}

inline void init(unsigned flags = 0) { checkCudaCall(hipInit(flags)); }

inline int driverGetVersion() {
  int version{};
  checkCudaCall(hipDriverGetVersion(&version));
  return version;
}

inline void memcpyHtoD(hipDeviceptr_t dst, const void *src, size_t size) {
  // const_cast is a temp fix for https://github.com/ROCm/ROCm/issues/2977
  checkCudaCall(hipMemcpyHtoD(dst, const_cast<void *>(src), size));
}

inline void memcpyDtoH(void *dst, hipDeviceptr_t src, size_t size) {
  checkCudaCall(hipMemcpyDtoH(dst, src, size));
}

class Context;
class Stream;

template <typename T>
class Wrapper {
 public:
  // conversion to C-style T

  operator T() const { return _obj; }

  operator T() { return _obj; }

  bool operator==(const Wrapper<T> &other) { return _obj == other._obj; }

  bool operator!=(const Wrapper<T> &other) { return _obj != other._obj; }

 protected:
  Wrapper() = default;

  Wrapper(const Wrapper<T> &other) : _obj(other._obj), manager(other.manager) {}

  Wrapper(Wrapper<T> &&other)
      : _obj(other._obj), manager(std::move(other.manager)) {
    other._obj = 0;
  }

  explicit Wrapper(T &obj) : _obj(obj) {}

  T _obj{};
  std::shared_ptr<T> manager;
};

class Device : public Wrapper<hipDevice_t> {
 public:
  // Device Management

  explicit Device(int ordinal) { checkCudaCall(hipDeviceGet(&_obj, ordinal)); }

  struct CUdeviceArg {
  };  // int and hipDevice_t are the same type, but we need two constructors
  Device(CUdeviceArg, hipDevice_t device) : Wrapper(device) {}

  int getAttribute(hipDeviceAttribute_t attribute) const {
    int value{};
    checkCudaCall(hipDeviceGetAttribute(&value, attribute, _obj));
    return value;
  }

  template <hipDeviceAttribute_t attribute>
  int getAttribute() const {
    return getAttribute(attribute);
  }

  static int getCount() {
    int nrDevices{};
    checkCudaCall(hipGetDeviceCount(&nrDevices));
    return nrDevices;
  }

  std::string getName() const {
    const size_t max_device_name_length{64};
    std::array<char, max_device_name_length> name{};
    checkCudaCall(hipDeviceGetName(name.data(), name.size(), _obj));
    return {name.data()};
  }

  std::string getUuid() const {
    hipUUID uuid;
    checkCudaCall(hipDeviceGetUuid(&uuid, _obj));

    // Convert a CUuuid to CUDA's string representation.
    // The CUuuid contains an array of 16 bytes, the UUID has
    // the form 'GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXX', with every
    // X being an alphanumeric character.
    std::stringstream result;
    result << "GPU";

    for (int i = 0; i < 16; ++i) {
      if (i == 0 || i == 4 || i == 6 || i == 8 || i == 10) {
        result << "-";
      }
      result << std::hex << std::setfill('0') << std::setw(2)
             << static_cast<unsigned>(
                    static_cast<unsigned char>(uuid.bytes[i]));
    }

    return result.str();
  }

  size_t totalMem() const {
    size_t size{};
    checkCudaCall(hipDeviceTotalMem(&size, _obj));
    return size;
  }

  // Primary Context Management

  std::pair<unsigned, bool> primaryCtxGetState() const {
    unsigned flags{};
    int active{};
    checkCudaCall(hipDevicePrimaryCtxGetState(_obj, &flags, &active));
    return {flags, active};
  }

  // void primaryCtxRelease() not available; it is released on destruction of
  // the Context returned by Device::primaryContextRetain()

  void primaryCtxReset() { checkCudaCall(hipDevicePrimaryCtxReset(_obj)); }

  Context primaryCtxRetain();  // retain this context until the primary context
                               // can be released

  void primaryCtxSetFlags(unsigned flags) {
    checkCudaCall(hipDevicePrimaryCtxSetFlags(_obj, flags));
  }
};

class Context : public Wrapper<hipCtx_t> {
 public:
  // Context Management

  Context(int flags, Device &device) : _primaryContext(false) {
    checkCudaCall(hipCtxCreate(&_obj, flags, device));
    manager = std::shared_ptr<hipCtx_t>(new hipCtx_t(_obj), [](hipCtx_t *ptr) {
      if (*ptr) hipCtxDestroy(*ptr);
      delete ptr;
    });
  }

  explicit Context(hipCtx_t context)
      : Wrapper<hipCtx_t>(context), _primaryContext(false) {}

  //  unsigned getApiVersion() const {
  //    unsigned version{};
  //    checkCudaCall(hipCtxGetApiVersion(_obj, &version));
  //    return version;
  //  }

  // Fix of cudawrappers by loostrum for HIP.
  unsigned getApiVersion() const {
    int version{};
    checkCudaCall(hipCtxGetApiVersion(_obj, &version));
    return static_cast<unsigned>(version);
  }

  static hipFuncCache_t getCacheConfig() {
    hipFuncCache_t config{};
    checkCudaCall(hipCtxGetCacheConfig(&config));
    return config;
  }

  int occupancyMaxActiveBlocksPerMultiprocessor(int blockSize,
                                                size_t dynamicSMemSize) {
    int numBlocks;
    checkCudaCall(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, _obj, blockSize, dynamicSMemSize));
    return numBlocks;
  }

  static void setCacheConfig(hipFuncCache_t config) {
    checkCudaCall(hipCtxSetCacheConfig(config));
  }

  static Context getCurrent() {
    hipCtx_t context{};
    checkCudaCall(hipCtxGetCurrent(&context));
    return Context(context);
  }

  void setCurrent() const { checkCudaCall(hipCtxSetCurrent(_obj)); }

  void pushCurrent() { checkCudaCall(hipCtxPushCurrent(_obj)); }

  static Context popCurrent() {
    hipCtx_t context{};
    checkCudaCall(hipCtxPopCurrent(&context));
    return Context(context);
  }

  static Device getDevice() {
    hipDevice_t device;
    checkCudaCall(hipCtxGetDevice(&device));
    return Device(Device::CUdeviceArg(), device);
  }

  static size_t getLimit(hipLimit_t limit) {
    size_t value{};
    checkCudaCall(hipDeviceGetLimit(&value, limit));
    return value;
  }

  template <hipLimit_t limit>
  static size_t getLimit() {
    return getLimit(limit);
  }

  static void setLimit(hipLimit_t limit, size_t value) {
    checkCudaCall(hipDeviceSetLimit(limit, value));
  }

  template <hipLimit_t limit>
  static void setLimit(size_t value) {
    setLimit(limit, value);
  }

  size_t getFreeMemory() const {
    size_t free;
    size_t total;
    checkCudaCall(hipMemGetInfo(&free, &total));
    return free;
  }

  size_t getTotalMemory() const {
    size_t free;
    size_t total;
    checkCudaCall(hipMemGetInfo(&free, &total));
    return total;
  }

  static void synchronize() { checkCudaCall(hipCtxSynchronize()); }

 private:
  friend class Device;
  Context(hipCtx_t context, Device &device)
      : Wrapper<hipCtx_t>(context), _primaryContext(true) {}

  bool _primaryContext;
};

class HostMemory : public Wrapper<void *> {
 public:
  explicit HostMemory(size_t size, unsigned int flags = 0) : _size(size) {
    checkCudaCall(hipHostAlloc(&_obj, size, flags));
    manager = std::shared_ptr<void *>(new (void *)(_obj), [](void **ptr) {
      checkCudaCall(hipHostFree(*ptr));
      delete ptr;
    });
  }

  explicit HostMemory(void *ptr, size_t size, unsigned int flags = 0)
      : _size(size) {
    _obj = ptr;
    checkCudaCall(hipHostRegister(_obj, size, flags));
    manager = std::shared_ptr<void *>(new (void *)(_obj), [](void **ptr) {
      checkCudaCall(hipHostUnregister(*ptr));
    });
  }

  template <typename T>
  operator T *() {
    return static_cast<T *>(_obj);
  }

  size_t size() const { return _size; }

 private:
  size_t _size;
};

class Array : public Wrapper<hipArray_t> {
 public:
  Array(unsigned width, hipArray_Format format, unsigned numChannels) {
    create2DArray(width, 0, format, numChannels);
  }

  Array(unsigned width, unsigned height, hipArray_Format format,
        unsigned numChannels) {
    create2DArray(width, height, format, numChannels);
  }

  Array(unsigned width, unsigned height, unsigned depth, hipArray_Format format,
        unsigned numChannels) {
    HIP_ARRAY3D_DESCRIPTOR descriptor;
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Depth = depth;
    descriptor.Format = format;
    descriptor.NumChannels = numChannels;
    descriptor.Flags = 0;
    checkCudaCall(hipArray3DCreate(&_obj, &descriptor));
    createManager();
  }

  explicit Array(hipArray_t &array) : Wrapper(array) {}

 private:
  void create2DArray(unsigned width, unsigned height, hipArray_Format format,
                     unsigned numChannels) {
    HIP_ARRAY_DESCRIPTOR descriptor;
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = numChannels;
    checkCudaCall(hipArrayCreate(&_obj, &descriptor));
    createManager();
  }

  void createManager() {
    manager =
        std::shared_ptr<hipArray_t>(new hipArray_t(_obj), [](hipArray_t *ptr) {
          hipArrayDestroy(*ptr);
          delete ptr;
        });
  }
};

class Module : public Wrapper<hipModule_t> {
 public:
  explicit Module(const char *file_name) {
#if defined TEGRA_QUIRKS  // hipModuleLoad broken on Jetson TX1
    std::ifstream file(file_name);
    std::string program((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    checkCudaCall(hipModuleLoadData(&_obj, program.c_str()));
#else
    checkCudaCall(hipModuleLoad(&_obj, file_name));
#endif
    manager = std::shared_ptr<hipModule_t>(new hipModule_t(_obj),
                                           [](hipModule_t *ptr) {
                                             hipModuleUnload(*ptr);
                                             delete ptr;
                                           });
  }

  explicit Module(const void *data) {
    checkCudaCall(hipModuleLoadData(&_obj, data));
    manager = std::shared_ptr<hipModule_t>(new hipModule_t(_obj),
                                           [](hipModule_t *ptr) {
                                             hipModuleUnload(*ptr);
                                             delete ptr;
                                           });
  }

  typedef std::map<hipJitOption, void *> optionmap_t;
  explicit Module(const void *image, Module::optionmap_t &options) {
    std::vector<hipJitOption> keys;
    std::vector<void *> values;

    for (const std::pair<hipJitOption, void *> &i : options) {
      keys.push_back(i.first);
      values.push_back(i.second);
    }

    checkCudaCall(hipModuleLoadDataEx(&_obj, image, options.size(), keys.data(),
                                      values.data()));

    for (size_t i = 0; i < keys.size(); ++i) {
      options[keys[i]] = values[i];
    }
  }

  explicit Module(hipModule_t &module) : Wrapper(module) {}

  hipDeviceptr_t getGlobal(const char *name) const {
    hipDeviceptr_t deviceptr{};
    checkCudaCall(hipModuleGetGlobal(&deviceptr, nullptr, _obj, name));
    return deviceptr;
  }
};

class Function : public Wrapper<hipFunction_t> {
 public:
  Function(const Module &module, const char *name) : _name(name) {
    checkCudaCall(hipModuleGetFunction(&_obj, module, name));
  }

  explicit Function(hipFunction_t &function) : Wrapper(function) {}

  int getAttribute(hipFunction_attribute attribute) const {
    int value{};
    checkCudaCall(hipFuncGetAttribute(&value, attribute, _obj));
    return value;
  }

  void setAttribute(hipFuncAttribute attribute, int value) {
    checkCudaCall(hipFuncSetAttribute(_obj, attribute, value));
  }

  void setCacheConfig(hipFuncCache_t config) {
    checkCudaCall(hipFuncSetCacheConfig(_obj, config));
  }

  const char *name() const { return _name; }

 private:
  const char *_name;
};

class Event : public Wrapper<hipEvent_t> {
 public:
  explicit Event(unsigned int flags = hipEventDefault) {
    checkCudaCall(hipEventCreateWithFlags(&_obj, flags));
    manager =
        std::shared_ptr<hipEvent_t>(new hipEvent_t(_obj), [](hipEvent_t *ptr) {
          hipEventDestroy(*ptr);
          delete ptr;
        });
  }

  explicit Event(hipEvent_t &event) : Wrapper(event) {}

  float elapsedTime(const Event &start) const {
    float ms{};
    checkCudaCall(hipEventElapsedTime(&ms, start, _obj));
    return ms;
  }

  void query() const {
    checkCudaCall(hipEventQuery(_obj));  // unsuccessful result throws cu::Error
  }

  void record() { checkCudaCall(hipEventRecord(_obj, 0)); }

  void record(Stream &);

  void synchronize() { checkCudaCall(hipEventSynchronize(_obj)); }
};

class DeviceMemory : public Wrapper<hipDeviceptr_t> {
 public:
  explicit DeviceMemory(size_t size, hipMemoryType type = hipMemoryTypeDevice,
                        unsigned int flags = 0)
      : _size(size) {
    if (size == 0) {
      _obj = 0;
      return;
    } else if (type == hipMemoryTypeDevice && !flags) {
      checkCudaCall(hipMalloc(&_obj, size));
    } else if (type == hipMemoryTypeUnified) {
      checkCudaCall(hipMallocManaged(&_obj, size, flags));
    } else {
      throw Error(hipErrorInvalidValue);
    }
    manager = std::shared_ptr<hipDeviceptr_t>(new hipDeviceptr_t(_obj),
                                              [](hipDeviceptr_t *ptr) {
                                                hipFree(*ptr);
                                                delete ptr;
                                              });
  }

  explicit DeviceMemory(hipDeviceptr_t ptr) : Wrapper(ptr) {}

  explicit DeviceMemory(hipDeviceptr_t ptr, size_t size)
      : Wrapper(ptr), _size(size) {}

  explicit DeviceMemory(const HostMemory &hostMemory) {
    checkCudaCall(hipHostGetDevicePointer(&_obj, hostMemory, 0));
  }

  void zero(size_t size) { checkCudaCall(hipMemsetD8(_obj, 0, size)); }

  const void *parameter()
      const  // used to construct parameter list for launchKernel();
  {
    return &_obj;
  }

  template <typename T>
  operator T *() {
    int data;
    checkCudaCall(
        hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_IS_MANAGED, _obj));
    if (data) {
      return reinterpret_cast<T *>(_obj);
    } else {
      throw std::runtime_error(
          "Cannot return memory of type hipMemoryTypeDevice as pointer.");
    }
  }

  size_t size() const { return _size; }

 private:
  size_t _size;
};

class Stream : public Wrapper<hipStream_t> {
  friend class Event;

 public:
  explicit Stream(unsigned int flags = hipStreamDefault) {
    checkCudaCall(hipStreamCreateWithFlags(&_obj, flags));
    manager = std::shared_ptr<hipStream_t>(new hipStream_t(_obj),
                                           [](hipStream_t *ptr) {
                                             hipStreamDestroy(*ptr);
                                             delete ptr;
                                           });
  }

  explicit Stream(hipStream_t stream) : Wrapper<hipStream_t>(stream) {}

  DeviceMemory memAllocAsync(size_t size) {
    hipDeviceptr_t ptr;
    checkCudaCall(hipMallocAsync(&ptr, size, _obj));
    return DeviceMemory(ptr, size);
  }

  void memFreeAsync(DeviceMemory &devMem) {
    checkCudaCall(hipFreeAsync(devMem, _obj));
  }

  void memcpyHtoHAsync(void *dstPtr, const void *srcPtr, size_t size) {
    checkCudaCall(hipMemcpyAsync(
        reinterpret_cast<hipDeviceptr_t>(dstPtr),
        reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(srcPtr)), size,
        hipMemcpyDefault, _obj));
  }

  void memcpyHtoDAsync(DeviceMemory &devPtr, const void *hostPtr, size_t size) {
    checkCudaCall(
        hipMemcpyHtoDAsync(devPtr, const_cast<void *>(hostPtr), size, _obj));
  }

  void memcpyHtoDAsync(hipDeviceptr_t devPtr, const void *hostPtr,
                       size_t size) {
    checkCudaCall(
        hipMemcpyHtoDAsync(devPtr, const_cast<void *>(hostPtr), size, _obj));
  }

  void memcpyDtoHAsync(void *hostPtr, const DeviceMemory &devPtr, size_t size) {
    checkCudaCall(hipMemcpyDtoHAsync(hostPtr, devPtr, size, _obj));
  }

  void memcpyDtoHAsync(void *hostPtr, hipDeviceptr_t devPtr, size_t size) {
    checkCudaCall(hipMemcpyDtoHAsync(hostPtr, devPtr, size, _obj));
  }

  void memcpyDtoDAsync(DeviceMemory &dstPtr, DeviceMemory &srcPtr,
                       size_t size) {
    checkCudaCall(hipMemcpyAsync(dstPtr, srcPtr, size, hipMemcpyDefault, _obj));
  }

  void memPrefetchAsync(DeviceMemory &devPtr, size_t size) {
    checkCudaCall(hipMemPrefetchAsync(devPtr, size, hipCpuDeviceId, _obj));
  }

  void memPrefetchAsync(DeviceMemory &devPtr, size_t size, Device &dstDevice) {
    checkCudaCall(hipMemPrefetchAsync(devPtr, size, dstDevice, _obj));
  }

  void zero(DeviceMemory &devPtr, size_t size) {
    checkCudaCall(hipMemsetD8Async(devPtr, 0, size, _obj));
  }

  void launchKernel(Function &function, unsigned gridX, unsigned gridY,
                    unsigned gridZ, unsigned blockX, unsigned blockY,
                    unsigned blockZ, unsigned sharedMemBytes,
                    const std::vector<const void *> &parameters) {
    checkCudaCall(hipModuleLaunchKernel(
        function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
        _obj, const_cast<void **>(&parameters[0]), nullptr));
  }

#if CUDART_VERSION >= 9000
  void launchCooperativeKernel(Function &function, unsigned gridX,
                               unsigned gridY, unsigned gridZ, unsigned blockX,
                               unsigned blockY, unsigned blockZ,
                               unsigned sharedMemBytes,
                               const std::vector<const void *> &parameters) {
    checkCudaCall(hipModuleLaunchCooperativeKernel(
        function, gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMemBytes,
        _obj, const_cast<void **>(&parameters[0])));
  }
#endif

  void query() {
    checkCudaCall(
        hipStreamQuery(_obj));  // unsuccessful result throws cu::Error
  }

  void synchronize() { checkCudaCall(hipStreamSynchronize(_obj)); }

  void wait(Event &event) { checkCudaCall(hipStreamWaitEvent(_obj, event, 0)); }

  void addCallback(hipStreamCallback_t callback, void *userData,
                   unsigned int flags = 0) {
    checkCudaCall(hipStreamAddCallback(_obj, callback, userData, flags));
  }

  void record(Event &event) { checkCudaCall(hipEventRecord(event, _obj)); }

  // void batchMemOp(unsigned count, CUstreamBatchMemOpParams *paramArray,
  //                 unsigned flags) {
  //   checkCudaCall(cuStreamBatchMemOp(_obj, count, paramArray, flags));
  // }

  void waitValue32(hipDeviceptr_t addr, cuuint32_t value,
                   unsigned flags) const {
    checkCudaCall(hipStreamWaitValue32(_obj, addr, value, flags));
  }

  void writeValue32(hipDeviceptr_t addr, cuuint32_t value, unsigned flags) {
    checkCudaCall(hipStreamWriteValue32(_obj, addr, value, flags));
  }

  //  Context getContext() const {
  //    hipCtx_t context;
  //    checkCudaCall(cuStreamGetCtx(_obj, &context));
  //    return Context(context);
  //  }
};

inline void Event::record(Stream &stream) {
  checkCudaCall(hipEventRecord(_obj, stream._obj));
}
}  // namespace cu

#endif
