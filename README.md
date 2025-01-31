[![github url](https://img.shields.io/badge/github-url-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/nlesc-recruit/cudawrappers)
[![github license badge](https://img.shields.io/github/license/nlesc-recruit/cudawrappers)](https://github.com/nlesc-recruit/cudawrappers)
[![DOI](https://zenodo.org/badge/424944643.svg)](https://zenodo.org/badge/latestdoi/424944643)
[![Research Software Directory](https://img.shields.io/badge/rsd-cudawrappers-00a3e3.svg)](https://www.research-software.nl/software/cudawrappers)
[![cii badge](https://bestpractices.coreinfrastructure.org/projects/5686/badge)](https://bestpractices.coreinfrastructure.org/projects/5686)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/d38b0338fda24733ab41a64915af8248)](https://app.codacy.com/gh/nlesc-recruit/cudawrappers/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nlesc-recruit/cudawrappers&amp;utm_campaign=Badge_Grade)
[![citation metadata](https://github.com/nlesc-recruit/cudawrappers/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/nlesc-recruit/cudawrappers/actions/workflows/cffconvert.yml)
[![Documentation Status](https://readthedocs.org/projects/cudawrappers/badge)](https://cudawrappers.readthedocs.io/en/latest/)


# cudawrappers

This library is a C++ wrapper for the Nvidia C libraries (e.g. CUDA driver, nvrtc, cuFFT etc.). The main purposes are:

1. _easier resource management_, leading to _lower risk of programming errors_;
2. _better fault handling_ (through exceptions);
3. _more compact user code_.

This library also supports AMD GPUs through the HIP: C++ Heterogeneous-Compute
Interface for Portability.

Originally, the API enforced RAII to even further reduce the risk of faulty code, but enforcing RAII and compatibility with (unmanaged) objects obtained outside this API are mutually exclusive.

## Requirements

| Software    | Minimum version |
| ----------- | ----------- |
| CUDA        | 10.0 or later |
| ROCM        | 6.1.0 or later |
| CMake       | 3.17 or later |
| gcc         | 9.3 or later  |
| OS          | Linux distro (amd64) |

| Hardware    | Type |
| ----------- | ----------- |
| NVIDIA GPU  |
[Pascal](https://www.nvidia.com/en-in/geforce/products/10series/architecture/)
or newer|
| AMD GPU     | RDNA2 or newer, CDNA2 or newer |



## Usage

CMake is used to build cudawrappers. To build the library, run:

```shell
git clone https://github.com/nlesc-recruit/cudawrappers
cd cudawrappers
cmake -S . -B build
make -C build
```

This creates a `build` directory. Since cudawrappers is header-only, no library objects will be built.
For more details on the building requirements and testing, see the [developer documentation](README.dev.md).

To install cudawrappers to a specific location (e.g., ~/.local), use:
```shell
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -S . -B build
make -C build
make -C build install
```

### Enabling HIP Support

To enable HIP:

1. Add the following code to your project's `CMakeLists.txt` file:

    ```cmake
    enable_language(HIP)
    set(CUDAWRAPPERS_BACKEND "HIP")
    ```

    Alternatively, use the `-DCUDAWRAPPERS_BACKEND=HIP` argument when running CMake.

2. Ensure that every target your project builds is 'hipfied'.
    This is done by marking relevant source files as 'HIP' compatible:

    ```cmake
    set_source_files_properties(source.cpp PROPERTIES LANGUAGE HIP)
    ```

3. Optionally, use `#ifdef (__HIP__)` directives in your source code to enable/disable certain sections for HIP.

4. **Build**: ensure that the `hipcc` compiler is selected.
    This can be done via the command line:

    ```shell
    CXX=hipcc cmake -B build
    ```

**Note**: When building for both NVIDIA and AMD HIP,
using **seperate** build folders (e.g, `build_nvidia` and `build_amd`) is encouraged.
Additionally, please note that contrary to CUDA, the HIP backend does not implement GPU contexts. For library interoperability, it provides a (non-functioning) mock implementation.
Still, the usage of `cu::Context` is strongly discouraged, and has been marked deprecated since cudawrappers version `0.9.0`. Support will likely be dropped in a future release.
### Code Examples

You can include the cudawrappers library in your own projects in various ways.
We have created a few repositories with example setups to get you started:

1. [cudawrappers-usage-example-git-submodules](https://github.com/nlesc-recruit/cudawrappers-usage-example-git-submodules) Example project that uses the cudawrappers library as a dependency by using git submodules on its source tree.
1. [cudawrappers-usage-example-locally-installed](https://github.com/nlesc-recruit/cudawrappers-usage-example-locally-installed) Example project that uses the cudawrappers library as a dependency by having it locally installed.
1. [cudawrappers-usage-example-cmake-pull](https://github.com/nlesc-recruit/cudawrappers-usage-example-cmake-pull) Example project that uses the cudawrappers library as a dependency by having cmake pull it in from github.

## Used by

This section aims to provide an overview of projects that use this repo's library (or something very similar), e.g. through git submodules or by including copies of this library in their source tree:

1. https://git.astron.nl/RD/dedisp/
1. https://git.astron.nl/RD/idg
1. https://git.astron.nl/RD/tensor-core-correlator

## Alternatives

This section provides an overview of similar tools in this space, and how they are different.

### [cuda-api-wrappers](https://github.com/eyalroz/cuda-api-wrappers)

- Aims to provide wrappers for the CUDA runtime API
- Development has slowed a bit recently
- Has 1 or 2 main developers
- Has gained quite a bit of attention (e.g. 440 stars; 57 forks)

The project is planning to support more of the Driver API (for fine-grained control of CUDA devices) and NVRTC API (for runtime compilation of kernels); there is a release candidate ([`v0.5.0-rc1`](https://github.com/eyalroz/cuda-api-wrappers/tree/v0.5.0-rc1)). It doesn't provide support for cuFFT and cuBLAS though.

### [cuda-wrapper](https://github.com/halmd-org/cuda-wrapper)

- Aims to provide a C++ wrapper for the CUDA Driver and Runtime APIs

### [CudaPlusPlus](https://github.com/apardyl/cudaplusplus)

- Aims to provide a C++ wrapper for the CUDA Driver API
- Project appears inactive

## Contributing

See [CONTRIBUTING.md](https://github.com/nlesc-recruit/cudawrappers/blob/main/CONTRIBUTING.md) for a guide on how to contribute.

## Developer documentation

See [README.dev.md](https://github.com/nlesc-recruit/cudawrappers/blob/main/README.dev.md) for documentation on setting up your development environment.
