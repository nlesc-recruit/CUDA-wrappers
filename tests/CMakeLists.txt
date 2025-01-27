include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

set(COMPONENTS cu nvrtc cufft vector_add graph)
if(${CUDAWRAPPERS_BACKEND_CUDA})
  list(APPEND COMPONENTS nvml)
endif()

foreach(component ${COMPONENTS})
  if(${CUDAWRAPPERS_BACKEND_HIP})
    set_source_files_properties(test_${component}.cpp PROPERTIES LANGUAGE HIP)
  endif()
  add_executable(test_${component} test_${component}.cpp)
  target_include_directories(
    test_${component} PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
  catch_discover_tests(test_${component})
endforeach()

set(LINK_LIBRARIES Catch2::Catch2WithMain cudawrappers::cu)

target_link_libraries(test_cu PUBLIC ${LINK_LIBRARIES})

if(${CUDAWRAPPERS_BACKEND_CUDA})
  target_link_libraries(test_nvml PUBLIC ${LINK_LIBRARIES} cudawrappers::nvml)
endif()

target_link_libraries(test_nvrtc PUBLIC ${LINK_LIBRARIES} cudawrappers::nvrtc)

target_embed_source(test_nvrtc kernels/vector_add_kernel.cu)

target_link_libraries(test_cufft PUBLIC ${LINK_LIBRARIES} cudawrappers::cufft)

target_link_libraries(
  test_vector_add PUBLIC ${LINK_LIBRARIES} cudawrappers::nvrtc
)

target_link_libraries(test_graph PUBLIC ${LINK_LIBRARIES} cudawrappers::nvrtc)
