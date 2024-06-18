# Change Log

All notable changes to this project will be documented in this file. This
project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Added

- Added `cu::Function::occupancyMaxActiveBlocksPerMultiprocessor()`
- Added `cu::Device::getUUID()`
- Added initial cudawrappers::nvml target
- Added `nvrtc::findIncludePath()`

### Changed

- `target_embed_source` will now automatically inline local header files

### Removed

- Removed deprecated `cu::Context::setSharedMemConfig`

## \[0.7.0\] - 2024-03-08

### Added

- Added `cu::Context::getDevice()`
- Added `cu::Module` constructor with `CUjit_option` map argument
- Added `cu::DeviceMemory::size`
- Added `cu::HostMemory::size`
- Added `cu::Function::name`
- Added `cu::Stream::getContext()`
- Added overloaded versions of `cu::Stream::memcpyDtoHAsync` and
  `cu::Stream::memcpyDtoHAsync` that take `CUdeviceptr` as an argument
- Added `cu::Function::setAttribute()`

### Changed

- Fixed the `cu::Module(CUmodule&)` constructor
- Added `cu::Function::getAttribute` is now const
- The `cu::DeviceMemory` constructor now works with `size == 0`

### Fixed

- Fix compatibility with C++20 and C++23
- Fix `cu::HostMemory` constructor for registered memory
- Fix `cu::DeviceMemory` operator `T *()` for managed memory
- Fix `cu::Stream::memAllocAsync` returns `cu::DeviceMemory` with initialized
  size

## \[0.6.0\] - 2023-10-06

### Changed

- Made the library header only
- Improved CMake configuration
- Moved asynchronous `::zero` from `cu::Device` to `cu::Stream`
- Replaced `include_cuda_code` helper with `target_embed_source`
- Changed some arguments from native to wrapped type

## \[0.5.0\] - 2023-09-25

### Added

- cufft wrappers for 1D and 2D complex-to-complex FFTs
- `cu::HostMemory` constructor for pre-allocated memory
- `cu::DeviceMemory` constructor for managed memory
- `cu::Stream::cuMemPrefetchAsync` for pre-fetching of managed memory
- `cu::Stream::memAllocAsync` and `cu::Stream::memFreeAsync`
- `cu::Context::getFreeMemory` and `cu::Context::getTotalMemory`

### Changed

- The `vector_add example` has now become a test
- Added `lib` prefix to shared libraries

### Removed

- `getDevice` function of `cu::Context`, use `cu::Device` constructor instead

## \[0.4.0\] - 2023-06-23

### Added

- CTest for testing
- nvtx library
- bump2version

### Changed

- Miscellaneous improvements to CMake and CI

### Removed

- `cu::Source` class. Use `nvrtc::Program` instead.
- Commented out code that was not used (anymore)

## \[0.3.0\] - 2022-03-08

### Added

- API documentation

### Changed

- Fixed build issues of `vector_add` example in `tests`
- Improved, linter rules

### Removed

- Moved usage examples to separate repositories

## \[0.2.0\] - 2022-03-02

### Added

- Several best practices were implemented, such as citation file, user and
  developer documentation, linters, formatters, pre-commit hooks, GitHub
  workflows, badges, and issue and pull request templates.

### Changed

- The name of the repository and the library are now `cudawrappers`.
- The folder structure has changed to better separate header and source files.

## \[0.1.0\] - 2022-02-14

### Added

- First release with existing code.
