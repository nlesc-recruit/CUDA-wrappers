#if !defined MACROS_H
#define MACROS_H
// translation of CUDA macros to HIP, only required when compiling for AMD
// see https://rocm.docs.amd.com/projects/HIPIFY/en/docs-5.4.0/tables/CUDA_Driver_API_functions_supported_by_HIP.html
#ifdef __HIP_PLATFORM_AMD__
#define CU_CTX_SCHED_BLOCKING_SYNC hipDeviceScheduleBlockingSync
#endif  // __HIP_PLATFORM_AMD__
#endif  // MACROS_H
