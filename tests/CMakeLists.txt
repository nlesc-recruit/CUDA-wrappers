include(FetchContent)

# We use Catch2 v2.x because it is the last version that supports C++11, see
# https://github.com/catchorg/Catch2.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.9
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

# set(COMPONENTS cu nvml nvrtc cufft vector_add)
set(COMPONENTS cu nvrtc cufft vector_add)

foreach(component ${COMPONENTS})
  set_source_files_properties(test_${component}.cpp PROPERTIES LANGUAGE HIP)
  add_executable(test_${component} test_${component}.cpp)
  target_include_directories(
    test_${component} PRIVATE "${CMAKE_SOURCE_DIR}/include"
  )
  catch_discover_tests(test_${component})
endforeach()

set(LINK_LIBRARIES Catch2::Catch2 cudawrappers::cu)

target_link_libraries(test_cu PUBLIC ${LINK_LIBRARIES})

# target_link_libraries(test_nvml PUBLIC ${LINK_LIBRARIES} cudawrappers::nvml)

target_link_libraries(test_nvrtc PUBLIC ${LINK_LIBRARIES} cudawrappers::nvrtc)

target_embed_source(test_nvrtc kernels/vector_add_kernel.cu)

target_link_libraries(test_cufft PUBLIC ${LINK_LIBRARIES} cudawrappers::cufft)

target_link_libraries(
  test_vector_add PUBLIC ${LINK_LIBRARIES} cudawrappers::nvrtc
)
