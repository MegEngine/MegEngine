if(NOT DEFINED HIP_PATH)
  if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH
        "/opt/rocm/hip"
        CACHE PATH "Path to which HIP has been installed")
  else()
    set(HIP_PATH
        $ENV{HIP_PATH}
        CACHE PATH "Path to which HIP has been installed")
  endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP QUIET)
if(HIP_FOUND)
  message(STATUS "Found HIP: " ${HIP_VERSION})
else()
  message(
    FATAL_ERROR
      "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location."
  )
endif()

if(${HIP_VERSION} VERSION_LESS 3.0)
  message(FATAL_ERROR "ROCM version needed 3. Please update ROCM.")
endif()

macro(hipconfig_get_option variable option)
  if(NOT DEFINED ${variable})
    execute_process(COMMAND ${HIP_HIPCONFIG_EXECUTABLE} ${option}
                    OUTPUT_VARIABLE ${variable})
  endif()
endmacro()

hipconfig_get_option(HIP_COMPILER "--compiler")
hipconfig_get_option(HIP_CPP_CONFIG "--cpp_config")

separate_arguments(HIP_CPP_CONFIG)

foreach(hip_config_item ${HIP_CPP_CONFIG})
  foreach(macro_name "__HIP_PLATFORM_HCC__" "__HIP_ROCclr__")
    if(${hip_config_item} STREQUAL "-D${macro_name}=")
      set(HIP_CPP_DEFINE "${HIP_CPP_DEFINE}#define ${macro_name}\n")
      set(HIP_CPP_UNDEFINE
          "${HIP_CPP_UNDEFINE}\
                    #ifdef ${macro_name}\n#undef ${macro_name}\n\
                    #else\n#error\n\
                    #endif\n")
    elseif(${hip_config_item} STREQUAL "-D${macro_name}")
      set(HIP_CPP_DEFINE "${HIP_CPP_DEFINE}#define ${macro_name} 1\n")
      set(HIP_CPP_UNDEFINE
          "${HIP_CPP_UNDEFINE}\
                    #ifdef ${macro_name}\n#undef ${macro_name}\n\
                    #else\n#error\n\
                    #endif\n")
    endif()
  endforeach()
endforeach()

message(STATUS "Using HIP compiler ${HIP_COMPILER}")

if(${HIP_COMPILER} STREQUAL "hcc")
  set(MGE_ROCM_LIBS hip_hcc)
  message(
    WARNING "hcc is not well supported, please modify link.txt to link with hipcc")
elseif(${HIP_COMPILER} STREQUAL "clang")
  set(MGE_ROCM_LIBS amdhip64)
endif()

list(APPEND MGE_ROCM_LIBS amdocl64 MIOpen rocblas rocrand)

set(HIP_INCLUDE_DIR ${HIP_ROOT_DIR}/../include)
set(HIP_LIBRARY_DIR ${HIP_ROOT_DIR}/../lib)

function(find_rocm_library name dirname include library)
  find_path(
    ${name}_LIBRARY_DIR
    NAMES ${library}
    HINTS "${${name}_ROOT_DIR}" "${HIP_ROOT_DIR}/../${dirname}"
    PATH_SUFFIXES lib lib/x86_64
    DOC "Path to ${name} library directory")

  if(${${name}_LIBRARY_DIR} MATCHES "NOTFOUND$")
    message(FATAL_ERROR "Can not find ${name} library")
  endif()

  find_path(
    ${name}_INCLUDE_DIR
    NAMES ${include}
    HINTS "${${name}_ROOT_DIR}" "${HIP_ROOT_DIR}/../${dirname}"
    PATH_SUFFIXES include
    DOC "Path to ${name} include directory")

  if(${name}_INCLUDE_DIR MATCHES "NOTFOUND$")
    message(FATAL_ERROR "Can not find ${name} include")
  endif()
  message(DEBUG "Found lib ${${name}_LIBRARY_DIR}, include ${${name}_INCLUDE_DIR}")
endfunction()

find_rocm_library(MIOPEN miopen miopen libMIOpen.so)
find_rocm_library(ROCBLAS rocblas rocblas.h librocblas.so)
find_rocm_library(ROCRAND rocrand rocrand.h librocrand.so)
find_rocm_library(AMDOCL opencl CL libamdocl64.so)
