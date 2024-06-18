# Copy the contents of the input file to the output file with all the local
# includes inlined. Local includes are assumed to have ""'s, e.g. having a line
# '#include "helper.h"` will lead to `helper.h` being inlined.
function(inline_local_includes input_file output_file)
  file(READ ${input_file} input_file_contents)
  set(include_regex "#include[ \t]*\"([^\"]+)\"")
  string(REGEX MATCHALL ${include_regex} includes ${input_file_contents})
  string(REGEX REPLACE ${include_regex} "" input_file_contents
                       "${input_file_contents}"
  )
  set(include_files "")
  foreach(include ${includes})
    string(REGEX REPLACE ${include_regex} "\\1" include ${include})
    file(GLOB_RECURSE INCLUDE_PATHS "${PROJECT_SOURCE_DIR}/*/${include}")
    list(SORT INCLUDE_PATHS ORDER DESCENDING)
    list(GET INCLUDE_PATHS 0 include_PATH)
    list(APPEND include_files ${include_PATH})
    file(READ ${include_PATH} include_contents)
    string(APPEND processed_file_contents "${include_contents}\n\n")
  endforeach()
  string(APPEND processed_file_contents "${input_file_contents}\n")
  file(WRITE ${output_file} "${processed_file_contents}")
  set(include_files
      ${include_files}
      PARENT_SCOPE
  )
endfunction()

# Make it possible to embed a source file in a library, and link it to a target.
# E.g. to link <kernel.cu> into target <example_program>, use
# target_embed_source(example_program, kernel.cu). This will expose symbols
# _binary_kernel_cu_start and _binary_kernel_cu_end.
function(target_embed_source target input_file)
  include(CMakeDetermineSystem)
  # Strip the path and extension from input_file
  get_filename_component(NAME ${input_file} NAME_WLE)
  # Get absolute path for input file
  get_filename_component(input_file_absolute ${input_file} ABSOLUTE)
  # Make a copy of the input file in the binary dir with inlined header files
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}"
                 input_file_inlined ${input_file_absolute}
  )
  inline_local_includes(${input_file_absolute} ${input_file_inlined})
  # Link the input_file into an object file
  string(REPLACE "${PROJECT_SOURCE_DIR}/" "" input_file_inlined_relative
                 ${input_file_absolute}
  )
  add_custom_command(
    OUTPUT ${NAME}.o
    COMMAND
      ld ARGS -r -b binary -A ${CMAKE_SYSTEM_PROCESSOR} -o
      "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o" ${input_file_inlined_relative}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${input_file} ${include_files}
    COMMENT "Creating object file for ${input_file}"
  )
  if(NOT TARGET ${NAME})
    # Create a proper static library for the .o file
    add_library(${NAME} STATIC ${NAME}.o)
    set_target_properties(${NAME} PROPERTIES LINKER_LANGUAGE CXX)
  endif()
  # Link the static library to the target
  target_link_libraries(${target} PRIVATE ${NAME})
endfunction()
