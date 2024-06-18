# cudawrappers requires the CUDA Toolkit.If you include cudawrappers in your
# project, you need to include the toolkit yourself set(CUDA_MIN_VERSION 10.0)
# if(${CUDAWRAPPERS_INSTALLED}) find_package(CUDAToolkit ${CUDA_MIN_VERSION}
# REQUIRED) else() if(${CUDAToolkit_FOUND}) if(${CUDAToolkit_VERSION_MAJOR} LESS
# ${CUDA_MIN_VERSION}) message(FATAL_ERROR "Insufficient CUDA version: "
# ${CUDAToolkit_VERSION} " < " ${CUDA_MIN_VERSION} ) endif() else() message(
# FATAL_ERROR "CUDAToolkit not found, use find_package(CUDAToolkit REQUIRED)." )
# endif() endif()

find_package(hip REQUIRED)
find_package(hipfft REQUIRED)
