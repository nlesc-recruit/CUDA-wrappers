# Get the include directory in the source tree
get_filename_component(
  CUDAWRAPPERS_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include ABSOLUTE
)

# Define all the individual components that cudawrappers provides
set(CUDAWRAPPERS_COMPONENTS cu cufft nvml nvrtc nvtx)
# set(LINK_cu CUDA::cuda_driver) set(LINK_cufft CUDA::cuda_driver CUDA::cufft)
# set(LINK_nvml CUDA::cuda_driver CUDA::nvml) set(LINK_nvrtc CUDA::cuda_driver
# CUDA::nvrtc) set(LINK_nvtx CUDA::nvToolsExt)

set(LINK_cu hip::host)
set(LINK_cufft hip::host hip::hipfft)
set(LINK_nvml hip::host)
set(LINK_nvrtc hip::host)
set(LINK_nvtx hip::host)

foreach(component ${CUDAWRAPPERS_COMPONENTS})
  add_library(${component} INTERFACE)
  # cudawrappers exposes targets like, cudawrappers::cu, so an alias is created
  add_library(${PROJECT_NAME}::${component} ALIAS ${component})
  target_link_libraries(${component} INTERFACE ${LINK_${component}})
  if(${CUDAWRAPPERS_INSTALLED})
    target_include_directories(
      ${component} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    set_target_properties(
      ${component}
      PROPERTIES PUBLIC_HEADER
                 ${CUDAWRAPPERS_INCLUDE_DIR}/cudawrappers/${component}.hpp
    )
  else()
    target_include_directories(
      ${component} INTERFACE ${CUDAWRAPPERS_INCLUDE_DIR}
    )
  endif()
endforeach()

# Install the header files and export the configuration
install(
  TARGETS ${CUDAWRAPPERS_COMPONENTS}
  EXPORT ${PROJECT_NAME}-targets
  COMPONENT ${PROJECT_NAME}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
