#.rst:
# FindBANG
# --------
#
# Tools for building BANG C files: libraries and build dependencies.
#
# This script locates the CAMBRICON BANG C tools.  It should work on linux,
# windows, and mac and should be reasonably up to date with BANG C
# releases.
#
# This script makes use of the standard find_package arguments of
# <VERSION>, REQUIRED and QUIET.  BANG_FOUND will report if an
# acceptable version of BANG was found.
#
# The following variables affect the behavior of the macros in the
# script (in alphebetical order).  Note that any of these flags can be
# changed multiple times in the same directory before calling
#
#   BANG_CNCC_FLAGS
#   -- Additional CNCC command line arguments.  NOTE: multiple arguments must be
#      semi-colon delimited (e.g. --bang-mlu-arch=xxx;-Wall)
#
#   BANG_TARGET_CPU_ARCH
#   -- Specify the name of the class of CPU architecture for which the input
#      files must be compiled (e.g. specify BANG_TARGET_CPU_ARCH=armv8a,
#      will append --target=armv8a to BANG_CNCC_FLAGS)
#
#   This code is licensed under the MIT License.  See the FindBANG.cmake script
#   for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved
###############################################################################

# FindBANG.cmake

###############################################################################
# Check for required components
###############################################################################
set(BANG_FOUND TRUE)

###############################################################################
# This macro helps us find the location of helper files we will need the full path to
###############################################################################
macro(BANG_FIND_HELPER_FILE _name _extension)
  set(_full_name "${_name}.${_extension}")
  # CMAKE_CURRENT_LIST_FILE contains the full path to the file currently being
  # processed.  Using this variable, we can pull out the current path, and
  # provide a way to get access to the other files we need local to here.
  get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
  set(BANG_${_name} "${CMAKE_CURRENT_LIST_DIR}/${_full_name}")
  if(NOT EXISTS "${BANG_${_name}}")
    set(error_message "${_full_name} not found in ${CMAKE_CURRENT_LIST_DIR}")
    if(BANG_FIND_REQUIRED)
      message(FATAL_ERROR "${error_message}")
    else()
      if(NOT BANG_FIND_QUIETLY)
        message(STATUS "${error_message}")
      endif()
    endif()
  endif()
  # Set this variable as internal, so the user isn't bugged with it.
  set(BANG_${_name} ${BANG_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

bang_find_helper_file(parse_cnbin cmake)
bang_find_helper_file(make2cmake cmake)
bang_find_helper_file(run_cncc cmake)

###############################################################################
# Add include directories to pass to the cncc command.
###############################################################################
macro(BANG_INCLUDE_DIRECTORIES)
  foreach(dir ${ARGN})
    list(APPEND BANG_CNCC_INCLUDE_ARGS_USER -I${dir})
  endforeach()
endmacro()

##############################################################################
# Separate the OPTIONS out from the sources
##############################################################################
macro(BANG_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

##############################################################################
# Parse the OPTIONS from ARGN and set the variables prefixed by _option_prefix
##############################################################################
macro(BANG_PARSE_CNCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${BANG_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()

#####################################################################
# BANG_INCLUDE_CNCC_DEPENDENCIES
# So we want to try and include the dependency file if it exists.  If
# it doesn't exist then we need to create an empty one, so we can
# include it.
# If it does exist, then we need to check to see if all the files it
# depends on exist.  If they don't then we should clear the dependency
# file and regenerate it later.  This covers the case where a header
# file has disappeared or moved.
#####################################################################
macro(BANG_INCLUDE_CNCC_DEPENDENCIES dependency_file)
  set(BANG_CNCC_DEPEND)
  set(BANG_CNCC_DEPEND_REGENERATE FALSE)

  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindBANG.cmake generated file.  Do not edit.\n")
  endif()
  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
  if(BANG_CNCC_DEPEND)
    #message("BANG_CNCC_DEPEND found")
    foreach(f ${BANG_CNCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(BANG_CNCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    #message("BANG_CNCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(BANG_CNCC_DEPEND_REGENERATE TRUE)
  endif()

  #message("BANG_CNCC_DEPEND_REGENERATE = ${BANG_CNCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(BANG_CNCC_DEPEND_REGENERATE)
    set(BANG_CNCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindBANG.cmake generated file.  Do not edit.\n")
  endif()

endmacro()

###############################################################################
# Helper to add the include directory for BANG only once
###############################################################################
function(BANG_ADD_BANG_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${BANG_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(${BANG_INCLUDE_DIRS})
  endif()
endfunction()

##############################################################################
# Build the shared library
##############################################################################
function(BANG_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _bang_found_SHARED)
  list(FIND cmake_args MODULE _bang_found_MODULE)
  list(FIND cmake_args STATIC _bang_found_STATIC)
  if( _bang_found_SHARED GREATER -1 OR
      _bang_found_MODULE GREATER -1 OR
      _bang_found_STATIC GREATER -1)
    set(_bang_build_shared_libs)
  else()
    if (BUILD_SHARED_LIBS)
      set(_bang_build_shared_libs SHARED)
    else()
      set(_bang_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_bang_build_shared_libs} PARENT_SCOPE)
endfunction()

##############################################################################
# Helper to avoid clashes of files with the same basename but different paths.
# This doesn't attempt to do exactly what CMake internals do, which is to only
# add this path when there is a conflict, since by the time a second collision
# in names is detected it's already too late to fix the first one.  For
# consistency sake the relative path will be added to all files.
##############################################################################
function(BANG_COMPUTE_BUILD_PATH path build_path)
  #message("BANG_COMPUTE_BUILD_PATH([${path}] ${build_path})")
  # Only deal with CMake style paths from here on out
  file(TO_CMAKE_PATH "${path}" bpath)
  if (IS_ABSOLUTE "${bpath}")
    # Absolute paths are generally unnessary, especially if something like
    # file(GLOB_RECURSE) is used to pick up the files.

    string(FIND "${bpath}" "${CMAKE_CURRENT_BINARY_DIR}" _binary_dir_pos)
    if (_binary_dir_pos EQUAL 0)
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_BINARY_DIR}" "${bpath}")
    else()
      file(RELATIVE_PATH bpath "${CMAKE_CURRENT_SOURCE_DIR}" "${bpath}")
    endif()
  endif()

  # This recipe is from cmLocalGenerator::CreateSafeUniqueObjectFileName in the
  # CMake source.

  # Remove leading /
  string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
  # Avoid absolute paths by removing ':'
  string(REPLACE ":" "_" bpath "${bpath}")
  # Avoid relative paths that go up the tree
  string(REPLACE "../" "__/" bpath "${bpath}")
  # Avoid spaces
  string(REPLACE " " "_" bpath "${bpath}")

  # Strip off the filename.  I wait until here to do it, since removin the
  # basename can make a path that looked like path/../basename turn into
  # path/.. (notice the trailing slash).
  get_filename_component(bpath "${bpath}" PATH)

  set(${build_path} "${bpath}" PARENT_SCOPE)
  #message("${build_path} = ${bpath}")
endfunction()

##############################################################################
# Compute the filename to be used by BANG_LINK_SEPARABLE_COMPILATION_OBJECTS
##############################################################################
function(BANG_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME output_file_var bang_target object_files)
  if (object_files)
    set(generated_extension ${CMAKE_${BANG_C_OR_CXX}_OUTPUT_EXTENSION})
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${bang_target}.dir/${CMAKE_CFG_INTDIR}/${bang_target}_intermediate_link${generated_extension}")
  else()
    set(output_file)
  endif()

  set(${output_file_var} "${output_file}" PARENT_SCOPE)
endfunction()

##############################################################################
# Setup the build rule for the separable compilation intermediate link file.
##############################################################################
function(BANG_LINK_SEPARABLE_COMPILATION_OBJECTS output_file bang_target options object_files)
  if (object_files)

    set_source_files_properties("${output_file}"
      PROPERTIES
      EXTERNAL_OBJECT TRUE # This is an object file not to be compiled, but only
                           # be linked.
      GENERATED TRUE       # This file is generated during the build
      )

    # For now we are ignoring all the configuration specific flags.
    set(cncc_flags)
    BANG_PARSE_CNCC_OPTIONS(cncc_flags ${options})

    # If -ccbin, --compiler-bindir has been specified, don't do anything.  Otherwise add it here.
    list( FIND cncc_flags "-ccbin" ccbin_found0 )
    list( FIND cncc_flags "--compiler-bindir" ccbin_found1 )
    if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND BANG_HOST_COMPILER )
      # Match VERBATIM check below.
      if(BANG_HOST_COMPILER MATCHES "\\$\\(VCInstallDir\\)")
        list(APPEND cncc_flags -ccbin "\"${BANG_HOST_COMPILER}\"")
      else()
        list(APPEND cncc_flags -ccbin "${BANG_HOST_COMPILER}")
      endif()
    endif()

    # Create a list of flags specified by BANG_CNCC_FLAGS_${CONFIG} and CMAKE_${BANG_C_OR_CXX}_FLAGS*
    set(config_specific_flags)
    set(flags)
    foreach(config ${BANG_configuration_types})
      string(TOUPPER ${config} config_upper)
      # Add config specific flags
      foreach(f ${BANG_CNCC_FLAGS_${config_upper}})
        list(APPEND config_specific_flags $<$<CONFIG:${config}>:${f}>)
      endforeach()
      set(important_host_flags)
      _bang_get_important_host_flags(important_host_flags "${CMAKE_${BANG_C_OR_CXX}_FLAGS_${config_upper}}")
      foreach(f ${important_host_flags})
        list(APPEND flags $<$<CONFIG:${config}>:-Xcompiler> $<$<CONFIG:${config}>:${f}>)
      endforeach()
    endforeach()
    # Add CMAKE_${BANG_C_OR_CXX}_FLAGS
    set(important_host_flags)
    _bang_get_important_host_flags(important_host_flags "${CMAKE_${BANG_C_OR_CXX}_FLAGS}")
    foreach(f ${important_host_flags})
      list(APPEND flags -Xcompiler ${f})
    endforeach()

    # Add our general BANG_CNCC_FLAGS with the configuration specifig flags
    set(cncc_flags ${BANG_CNCC_FLAGS} ${config_specific_flags} ${cncc_flags})

    file(RELATIVE_PATH output_file_relative_path "${CMAKE_BINARY_DIR}" "${output_file}")

    # Some generators don't handle the multiple levels of custom command
    # dependencies correctly (obj1 depends on file1, obj2 depends on obj1), so
    # we work around that issue by compiling the intermediate link object as a
    # pre-link custom command in that situation.
    set(do_obj_build_rule TRUE)
    if (MSVC_VERSION GREATER 1599 AND MSVC_VERSION LESS 1800)
      # VS 2010 and 2012 have this problem.
      set(do_obj_build_rule FALSE)
    endif()

    set(_verbatim VERBATIM)
    if(cncc_flags MATCHES "\\$\\(VCInstallDir\\)")
      set(_verbatim "")
    endif()

    if (do_obj_build_rule)
      add_custom_command(
        OUTPUT ${output_file}
        DEPENDS ${object_files}
        COMMAND ${BANG_CNCC_EXECUTABLE} ${cncc_flags} -dlink ${object_files} -o ${output_file}
        ${flags}
        COMMENT "Building CNCC intermediate link file ${output_file_relative_path}"
        ${_verbatim}
        )
    else()
      get_filename_component(output_file_dir "${output_file}" DIRECTORY)
      add_custom_command(
        TARGET ${bang_target}
        PRE_LINK
        COMMAND ${CMAKE_COMMAND} -E echo "Building CNCC intermediate link file ${output_file_relative_path}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${output_file_dir}"
        COMMAND ${BANG_CNCC_EXECUTABLE} ${cncc_flags} ${flags} -dlink ${object_files} -o "${output_file}"
        ${_verbatim}
        )
    endif()
 endif()
endfunction()

##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to invoke the cncc compiler to generate C or MLISA source
# dependent upon the format parameter.
# INPUT:
#   bang_target         - Target name
#   format              - MLISA, CNBIN, CNFATBIN or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to CNCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
macro(BANG_WRAP_SRCS bang_target format generated_files)
  # Set up all the command line flags here, so that they can be overridden on a per target basis.

  set(cncc_flags "")

  set(BANG_C_OR_CXX CXX)

  set(generated_extension ${CMAKE_${BANG_C_OR_CXX}_OUTPUT_EXTENSION})

  if(BANG_TARGET_CPU_ARCH)
    set(cncc_flags ${cncc_flags} "--target=${BANG_TARGET_CPU_ARCH}")
  endif()

  set( BANG_build_configuration "${CMAKE_BUILD_TYPE}")

  # Get the include directories for this directory and use them for our cncc command.
  # Remove duplicate entries which may be present since include_directories
  # in CMake >= 2.8.8 does not remove them.
  get_directory_property(BANG_CNCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES BANG_CNCC_INCLUDE_DIRECTORIES)
  if(BANG_CNCC_INCLUDE_DIRECTORIES)
    foreach(dir ${BANG_CNCC_INCLUDE_DIRECTORIES})
      list(APPEND BANG_CNCC_INCLUDE_ARGS -I${dir})
    endforeach()
  endif()

  # Reset these variables
  set(BANG_WRAP_OPTION_CNCC_FLAGS)
  foreach(config ${BANG_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(BANG_WRAP_OPTION_CNCC_FLAGS_${config_upper})
  endforeach()

  BANG_GET_SOURCES_AND_OPTIONS(_bang_wrap_sources _bang_wrap_cmake_options _bang_wrap_options ${ARGN})
  BANG_PARSE_CNCC_OPTIONS(BANG_WRAP_OPTION_CNCC_FLAGS ${_bang_wrap_options})

  # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
  # respected in BANG_ADD_LIBRARY.
  set(_bang_build_shared_libs FALSE)
  # SHARED, MODULE
  list(FIND _bang_wrap_cmake_options SHARED _bang_found_SHARED)
  list(FIND _bang_wrap_cmake_options MODULE _bang_found_MODULE)
  if(_bang_found_SHARED GREATER -1 OR _bang_found_MODULE GREATER -1)
    set(_bang_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _bang_wrap_cmake_options STATIC _bang_found_STATIC)
  if(_bang_found_STATIC GREATER -1)
    set(_bang_build_shared_libs FALSE)
  endif()

  # BANG_HOST_FLAGS
  if(_bang_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(BANG_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${BANG_C_OR_CXX}_FLAGS})
  else()
    set(BANG_HOST_SHARED_FLAGS)
  endif()
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  set(_bang_host_flags "set(CMAKE_HOST_FLAGS ${CMAKE_${BANG_C_OR_CXX}_FLAGS} ${BANG_HOST_SHARED_FLAGS})")

  set(_bang_cncc_flags_config "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_cncc.cmake
  foreach(config ${BANG_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we convert the strings to lists (like we want).

    set(_bang_C_FLAGS "${CMAKE_${BANG_C_OR_CXX}_FLAGS_${config_upper}}")
    set(_bang_host_flags "${_bang_host_flags}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_bang_C_FLAGS})")

    # Note that if we ever want BANG_CNCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is currently), we can remove the quotes around the
    # ${BANG_CNCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(_bang_cncc_flags_config "${_bang_cncc_flags_config}\nset(BANG_CNCC_FLAGS_${config_upper} ${BANG_CNCC_FLAGS_${config_upper}} ;; ${BANG_WRAP_OPTION_CNCC_FLAGS_${config_upper}})")
  endforeach()

  # Process the C++11 flag.  If the host sets the flag, we need to add it to cncc and
  # remove it from the host. This is because -Xcompile -std=c++ will choke cncc (it uses
  # the C preprocessor).  In order to get this to work correctly, we need to use cncc's
  # specific c++11 flag.
  if( "${_bang_host_flags}" MATCHES "-std=c\\+\\+11")
    # Add the c++11 flag to cncc if it isn't already present.  Note that we only look at
    # the main flag instead of the configuration specific flags.
    if( NOT "${BANG_CNCC_FLAGS}" MATCHES "-std;c\\+\\+11" )
      list(APPEND cncc_flags -std=c++11)
    endif()
    string(REGEX REPLACE "[-]+std=c\\+\\+11" "" _bang_host_flags "${_bang_host_flags}")
  endif()

  # Get the list of definitions from the directory property
  get_directory_property(BANG_CNCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(BANG_CNCC_DEFINITIONS)
    foreach(_definition ${BANG_CNCC_DEFINITIONS})
      list(APPEND cncc_flags "-D${_definition}")
    endforeach()
  endif()

  if(_bang_build_shared_libs)
    list(APPEND cncc_flags "-D${bang_target}_EXPORTS")
  endif()

  # Reset the output variable
  set(_bang_wrap_generated_files "")

  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  foreach(file ${ARGN})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    # Allow per source file overrides of the format.  Also allows compiling non-.cu files.
    get_source_file_property(_bang_source_format ${file} BANG_SOURCE_PROPERTY_FORMAT)
    if((${file} MATCHES "\\.mlu$" OR _bang_source_format) AND NOT _is_header)

      if(NOT _bang_source_format)
        set(_bang_source_format ${format})
      endif()

      if( ${_bang_source_format} MATCHES "OBJ")
        set( bang_compile_to_external_module OFF )
      else()
        set( bang_compile_to_external_module ON )
        if( ${_bang_source_format} MATCHES "MLISA" )
          set( bang_compile_to_external_module_type "mlisa" )
        elseif( ${_bang_source_format} MATCHES "CNBIN")
          set( bang_compile_to_external_module_type "cnbin" )
        elseif( ${_bang_source_format} MATCHES "CNFATBIN")
          set( bang_compile_to_external_module_type "cnfatbin" )
        else()
          message( FATAL_ERROR "Invalid format flag passed to BANG_WRAP_SRCS or set with BANG_SOURCE_PROPERTY_FORMAT file property for file '${file}': '${_bang_source_format}'.  Use OBJ, MLISA, CNBIN or CNFATBIN.")
        endif()
      endif()

      if(bang_compile_to_external_module)
        # Don't use any of the host compilation flags for MLISA targets.
        set(BANG_HOST_FLAGS)
        set(BANG_CNCC_FLAGS_CONFIG)
      else()
        set(BANG_HOST_FLAGS ${_bang_host_flags})
        set(BANG_CNCC_FLAGS_CONFIG ${_bang_cncc_flags_config})
      endif()

      # Determine output directory
      bang_compute_build_path("${file}" bang_build_path)
      set(bang_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${bang_target}.dir/${bang_build_path}")
      if(BANG_GENERATED_OUTPUT_DIR)
        set(bang_compile_output_dir "${BANG_GENERATED_OUTPUT_DIR}")
      else()
        if ( bang_compile_to_external_module )
          set(bang_compile_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
        else()
          set(bang_compile_output_dir "${bang_compile_intermediate_directory}")
        endif()
      endif()

      # Add a custom target to generate a c or mlisa file. ######################

      get_filename_component( basename ${file} NAME )
      if( bang_compile_to_external_module )
        set(generated_file_path "${bang_compile_output_dir}")
        set(generated_file_basename "${bang_target}_generated_${basename}.${bang_compile_to_external_module_type}")
        set(format_flag "-${bang_compile_to_external_module_type}")
        file(MAKE_DIRECTORY "${bang_compile_output_dir}")
      else()
        set(generated_file_path "${bang_compile_output_dir}/${CMAKE_CFG_INTDIR}")
        set(generated_file_basename "${bang_target}_generated_${basename}${generated_extension}")
        set(format_flag "-c")
      endif()

      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${bang_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(CNCC_generated_dependency_file "${bang_compile_intermediate_directory}/${generated_file_basename}.CNCC-depend")
      set(generated_cnbin_file "${generated_file_path}/${generated_file_basename}.cnbin.txt")
      set(custom_target_script "${bang_compile_intermediate_directory}/${generated_file_basename}.cmake")

      # Setup properties for obj files:
      if( NOT bang_compile_to_external_module )
        set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
      endif()

      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()

      # Bring in the dependencies.  Creates a variable BANG_CNCC_DEPEND #######
      bang_include_cncc_dependencies(${cmake_dependency_file})

      # Build the CNCC made dependency file ###################################
      set(build_cnbin OFF)
      if ( BANG_BUILD_CNBIN )
         if ( NOT bang_compile_to_external_module )
           set ( build_cnbin ON )
         endif()
      endif()

      # Configure the build script
      configure_file("${BANG_run_cncc}" "${custom_target_script}" @ONLY)

      # So if a user specifies the same bang file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      set(main_dep DEPENDS ${source_file})

      if(BANG_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()

      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      if(bang_compile_to_external_module)
        set(bang_build_comment_string "Building CNCC ${bang_compile_to_external_module_type} file ${generated_file_relative_path}")
      else()
        set(bang_build_comment_string "Building CNCC object ${generated_file_relative_path}")
      endif()

      set(_verbatim VERBATIM)
      if(ccbin_flags MATCHES "\\$\\(VCInstallDir\\)")
        set(_verbatim "")
      endif()

      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${BANG_CNCC_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${BANG_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_cnbin_file:STRING=${generated_cnbin_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${bang_compile_intermediate_directory}"
        COMMENT "${bang_build_comment_string}"
        ${_verbatim}
        )

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      list(APPEND _bang_wrap_generated_files ${generated_file})

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND BANG_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES BANG_ADDITIONAL_CLEAN_FILES)
      set(BANG_ADDITIONAL_CLEAN_FILES ${BANG_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the bang dependency scanning.")

    endif()
  endforeach()

  # Set the return parameter
  set(${generated_files} ${_bang_wrap_generated_files})
endmacro()

###############################################################################
###############################################################################
# Locate NEUWARE, Set Build Type, etc.
###############################################################################
###############################################################################

# BANG_CNCC_EXECUTABLE
find_program(BANG_CNCC_EXECUTABLE
  NAMES cncc
  PATHS "${NEUWARE_ROOT_DIR}"
  ENV NEUWARE_PATH
  ENV NEUWARE_BIN_PATH
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_program(BANG_CNCC_EXECUTABLE cncc)
mark_as_advanced(BANG_CNCC_EXECUTABLE)

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(BANG_ADD_LIBRARY bang_target)

  BANG_ADD_BANG_INCLUDE_ONCE()

  # Separate the sources from the options
  BANG_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  BANG_BUILD_SHARED_LIBRARY(_bang_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  BANG_WRAP_SRCS( ${bang_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_bang_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  BANG_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${bang_target} "${${bang_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${bang_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${bang_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  BANG_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${bang_target} "${_options}" "${${bang_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${bang_target}
    ${BANG_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. BANG_C_OR_CXX is computed based on BANG_HOST_COMPILATION_CPP.
  set_target_properties(${bang_target}
    PROPERTIES
    LINKER_LANGUAGE ${BANG_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# ADD EXECUTABLE
###############################################################################
###############################################################################
macro(BANG_ADD_EXECUTABLE bang_target)

  BANG_ADD_BANG_INCLUDE_ONCE()

  # Separate the sources from the options
  BANG_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  BANG_WRAP_SRCS( ${bang_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  BANG_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${bang_target} "${${bang_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_executable(${bang_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${bang_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  BANG_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${bang_target} "${_options}" "${${bang_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${bang_target}
    ${BANG_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. BANG_C_OR_CXX is computed based on BANG_HOST_COMPILATION_CPP.
  set_target_properties(${bang_target}
    PROPERTIES
    LINKER_LANGUAGE ${BANG_C_OR_CXX}
    )

endmacro()

#####################################################################################
# This macro invokes the cncc compiler to compile mlu source files, and to generate 
# object files.
# INPUT:
#   _source_files                          - List of mlu source files to be compiled
#   _include_directories_generator_user    - List of user provided include files
# OUTPUT:
#   _generated_files                       - List of generated files
#####################################################################################
macro(BANG_COMPILE _generated_files _source_files _include_directories_generator_user)
  # Set up all the command line flags here, so that they can be overridden on a per target basis.
  set(cncc_flags "")
  
  set(BANG_C_OR_CXX CXX)
  
  set(generated_extension ${CMAKE_${BANG_C_OR_CXX}_OUTPUT_EXTENSION})
  
  if(BANG_TARGET_CPU_ARCH)
    set(cncc_flags ${cncc_flags} "--target=${BANG_TARGET_CPU_ARCH}")
  endif()
  
  set( BANG_build_configuration "${CMAKE_BUILD_TYPE}")
  
  set(include_directories_generator ${_include_directories_generator_user})
  foreach(dir ${include_directories_generator})
    if(dir)
      list(APPEND BANG_CNCC_INCLUDE_ARGS -I${dir})
    endif()
  endforeach()
  
  # Reset these variables
  set(BANG_WRAP_OPTION_CNCC_FLAGS)
  foreach(config ${BANG_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(BANG_WRAP_OPTION_CNCC_FLAGS_${config_upper})
  endforeach()
  
  BANG_GET_SOURCES_AND_OPTIONS(_bang_wrap_sources _bang_wrap_cmake_options _bang_wrap_options ${_source_files})
  BANG_PARSE_CNCC_OPTIONS(BANG_WRAP_OPTION_CNCC_FLAGS ${_bang_wrap_options})
  
  set(BANG_HOST_SHARED_FLAGS)
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  set(_bang_host_flags "set(CMAKE_HOST_FLAGS ${CMAKE_${BANG_C_OR_CXX}_FLAGS} ${BANG_HOST_SHARED_FLAGS})")
  
  set(_bang_cncc_flags_config "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_cncc.cmake
  foreach(config ${BANG_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we convert the strings to lists (like we want).
  
    set(_bang_C_FLAGS "${CMAKE_${BANG_C_OR_CXX}_FLAGS_${config_upper}}")
    set(_bang_host_flags "${_bang_host_flags}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_bang_C_FLAGS})")
  
    # Note that if we ever want BANG_CNCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is currently), we can remove the quotes around the
    # ${BANG_CNCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(_bang_cncc_flags_config "${_bang_cncc_flags_config}\nset(BANG_CNCC_FLAGS_${config_upper} ${BANG_CNCC_FLAGS_${config_upper}} ;; ${BANG_WRAP_OPTION_CNCC_FLAGS_${config_upper}})")
  endforeach()
  
  # Process the C++11 flag.  If the host sets the flag, we need to add it to cncc and
  # remove it from the host. This is because -Xcompile -std=c++ will choke cncc (it uses
  # the C preprocessor).  In order to get this to work correctly, we need to use cncc's
  # specific c++11 flag.
  if( "${_bang_host_flags}" MATCHES "-std=c\\+\\+11")
    # Add the c++11 flag to cncc if it isn't already present.  Note that we only look at
    # the main flag instead of the configuration specific flags.
    if( NOT "${BANG_CNCC_FLAGS}" MATCHES "-std;c\\+\\+11" )
      list(APPEND cncc_flags -std=c++11)
    endif()
    string(REGEX REPLACE "[-]+std=c\\+\\+11" "" _bang_host_flags "${_bang_host_flags}")
  endif()
  
  # Get the list of definitions from the directory property
  get_directory_property(BANG_CNCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(BANG_CNCC_DEFINITIONS)
    foreach(_definition ${BANG_CNCC_DEFINITIONS})
      list(APPEND cncc_flags "-D${_definition}")
    endforeach()
  endif()
  
  # Reset the output variable
  set(_bang_wrap_generated_files "")
  
  # Iterate over the macro arguments and create custom
  # commands for all the .cu files.
  foreach(file ${_source_files})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    # Allow per source file overrides of the format.  Also allows compiling non-.cu files.
    get_source_file_property(_bang_source_format ${file} BANG_SOURCE_PROPERTY_FORMAT)
    if((${file} MATCHES "\\.mlu$" OR _bang_source_format) AND NOT _is_header)
  
      set(_bang_source_format "OBJ")
  
      set(BANG_HOST_FLAGS ${_bang_host_flags})
      set(BANG_CNCC_FLAGS_CONFIG ${_bang_cncc_flags_config})
  
      # Determine output directory
      bang_compute_build_path("${file}" bang_build_path)
      set(bang_compile_intermediate_directory "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/bang_compile.dir/${bang_build_path}")
      if(BANG_GENERATED_OUTPUT_DIR)
        set(bang_compile_output_dir "${BANG_GENERATED_OUTPUT_DIR}")
      else()
        set(bang_compile_output_dir "${bang_compile_intermediate_directory}")
      endif()
  
      # Add a custom target to generate a c or mlisa file. ######################
  
      get_filename_component( basename ${file} NAME )
      set(generated_file_path "${bang_compile_output_dir}/${CMAKE_CFG_INTDIR}")
      set(generated_file_basename "bang_compile_generated_${basename}${generated_extension}")
      set(format_flag "-c")
  
      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${bang_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(CNCC_generated_dependency_file "${bang_compile_intermediate_directory}/${generated_file_basename}.CNCC-depend")
      set(generated_cnbin_file "${generated_file_path}/${generated_file_basename}.cnbin.txt")
      set(custom_target_script "${bang_compile_intermediate_directory}/${generated_file_basename}.cmake")
  
      set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
  
      # Don't add CMAKE_CURRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
      endif()
  
      # Bring in the dependencies.  Creates a variable BANG_CNCC_DEPEND #######
      bang_include_cncc_dependencies(${cmake_dependency_file})
  
      # Build the CNCC made dependency file ###################################
      set(build_cnbin OFF)
  
      # Configure the build script
      configure_file("${BANG_run_cncc}" "${custom_target_script}" @ONLY)
  
      # So if a user specifies the same bang file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      set(main_dep DEPENDS ${source_file})
  
      if(BANG_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()
  
      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      set(bang_build_comment_string "Building CNCC object ${generated_file_relative_path}")
  
      set(_verbatim VERBATIM)
      if(ccbin_flags MATCHES "\\$\\(VCInstallDir\\)")
        set(_verbatim "")
      endif()
  
      # Build the generated file and dependency file ##########################
      add_custom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${BANG_CNCC_DEPEND}
        DEPENDS ${custom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${BANG_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_cnbin_file:STRING=${generated_cnbin_file}"
          -P "${custom_target_script}"
        WORKING_DIRECTORY "${bang_compile_intermediate_directory}"
        COMMENT "${bang_build_comment_string}"
        ${_verbatim}
        )
  
      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)
  
      list(APPEND _bang_wrap_generated_files ${generated_file})
  
      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND BANG_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES BANG_ADDITIONAL_CLEAN_FILES)
      set(BANG_ADDITIONAL_CLEAN_FILES ${BANG_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the bang dependency scanning.")
  
    endif()
  endforeach()
  
  # Set the return parameter
  set(${_generated_files} ${_bang_wrap_generated_files})
endmacro()
