cmake_minimum_required(VERSION 3.15.2)

include (cmake/FetchMegBrainVersion.cmake)
project(MegEngine LANGUAGES C CXX VERSION ${MGB_VER_STRING})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

if(NOT MSVC AND NOT APPLE AND NOT WIN32)
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>")
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Wclass-memaccess CXX_SUPPORT_WCLASS_MEMACCESS)

set(MGE_ARCH AUTO CACHE STRING "Architecture on which MegEngine to be built.")
set_property(CACHE MGE_ARCH PROPERTY STRINGS AUTO
    x86_64 i386
    armv7 aarch64
    naive fallback
)
set (MGE_EXPORT_TARGETS MegEngine-targets)

option(MGE_WITH_JIT "Build MegEngine with JIT." ON)
option(MGE_WITH_JIT_MLIR "Build MegEngine with MLIR JIT." OFF)
option(MGE_WITH_HALIDE "Build MegEngine with Halide JIT" ON)
option(MGE_WITH_MIDOUT_PROFILE "Build MegEngine with Midout profile." OFF)
option(MGE_WITH_MINIMUM_SIZE "Swith off MGE_ENABLE_RTTI、MGE_ENABLE_EXCEPTIONS、MGE_ENABLE_LOGGING and switch on MGE_INFERENCE_ONLY so that compile minimum load_and_run. Take effect only when MGE_BIN_REDUCE was set" OFF)
option(MGE_ARMV8_2_FEATURE_FP16 "Enable armv8.2-a+fp16 support" OFF)
option(MGE_ARMV8_2_FEATURE_DOTPROD "enable armv8.2-a+dotprod support" OFF)
option(MGE_DISABLE_FLOAT16 "Disable MegEngine float16 support." OFF)
option(MGE_WITH_CUDA "Enable MegEngine CUDA support." ON)
option(MGE_CUDA_USE_STATIC "Enable MegEngine CUDA static linking." ON)
option(MGE_WITH_TRT "Build MegEngine with TensorRT." ON)
option(MGE_USE_SYSTEM_LIB "Build MegEngine with system libraries." OFF)
option(MGB_WITH_FLATBUFFERS "Build MegBrain with FlatBuffers serialization support." ON)
option(MGE_WITH_CAMBRICON "Build MegEngine with Cambricon support" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(MGE_WITH_ATLAS "Build MegEngine with Atlas support" OFF)
option(MGE_ENABLE_RTTI "Build with RTTI" ON)
option(MGE_ENABLE_LOGGING "Build with logging" ON)
option(MGE_DEBUG_UTIL "Enable debug utility" ON)
option(MGE_ENABLE_EXCEPTIONS "Build with exceptions" ON)
option(MGE_WITH_TEST "Enable test for MegEngine." OFF)
option(MGE_WITH_DISTRIBUTED "Build with distributed support" ON)
option(MGE_BUILD_IMPERATIVE_RT "Build _imperative_rt Python Module " ON)
option(MGE_BUILD_SDK "Build load_and_run" ON)
option(MGE_INFERENCE_ONLY "Build inference only library." OFF)
option(MGE_WITH_PYTHON_MODULE "Build MegEngine legacy Python Module." OFF)
option(MGE_WITH_MKLDNN "Enable Intel MKL_DNN support," ON)
option(MGE_WITH_ROCM "Enable ROCM support" OFF)


if(NOT ${MGE_BIN_REDUCE} STREQUAL "")
    message(STATUS "build with BIN REDUCE")
    if(MGE_WITH_MINIMUM_SIZE)
        set(MGE_ENABLE_RTTI OFF)
        set(MGE_ENABLE_LOGGING OFF)
        set(MGE_ENABLE_EXCEPTIONS OFF)
        set(MGE_INFERENCE_ONLY ON)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${MGE_BIN_REDUCE}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -include ${MGE_BIN_REDUCE}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=full")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=full")
endif()

if(MGE_WITH_MIDOUT_PROFILE)
    message(STATUS "build with MIDOUT PROFILE")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIDOUT_PROFILING")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIDOUT_PROFILING")
endif()

if (APPLE)
    set (BUILD_SHARED_LIBS OFF)
    message(STATUS "build static for xcode framework require")
endif()

if (MGE_USE_SYSTEM_LIB)
    set (MGE_CUDA_USE_STATIC OFF)
endif()

if (MGB_WITH_FLATBUFFERS)
    set(MGB_ENABLE_FBS_SERIALIZATION ON)
endif()

if(CMAKE_TOOLCHAIN_FILE)
    message(STATUS "We are cross compiling.")
    message(STATUS "config FLATBUFFERS_FLATC_EXECUTABLE to: ${PROJECT_SOURCE_DIR}/build_dir/host_flatc/install/bin/flatc")
    set(FLATBUFFERS_FLATC_EXECUTABLE "${PROJECT_SOURCE_DIR}/build_dir/host_flatc/install/bin/flatc")
    if(ANDROID_TOOLCHAIN_ROOT)
        if(NOT "${ANDROID_ARCH_NAME}" STREQUAL "")
            set(ANDROID_ARCH ${ANDROID_ARCH_NAME})
        endif()
        if(${ANDROID_ARCH} STREQUAL "arm")
            set(MGE_ARCH "armv7")
        elseif(${ANDROID_ARCH} STREQUAL "arm64")
            set(MGE_ARCH "aarch64")
        else()
            message(FATAL_ERROR "DO NOT SUPPORT ANDROID ARCH NOW")
        endif()
    elseif(IOS_TOOLCHAIN_ROOT)
        if(${IOS_ARCH} STREQUAL "armv7")
            set(MGE_ARCH "armv7")
        elseif(${IOS_ARCH} STREQUAL "arm64")
            set(MGE_ARCH "aarch64")
        elseif(${IOS_ARCH} STREQUAL "armv7k")
            set(MGE_ARCH "armv7")
        elseif(${IOS_ARCH} STREQUAL "arm64e")
            set(MGE_ARCH "aarch64")
        elseif(${IOS_ARCH} STREQUAL "armv7s")
            set(MGE_ARCH "armv7")
        else()
            message(FATAL_ERROR "Unsupported IOS_ARCH.")
        endif()
    elseif(RISCV_TOOLCHAIN_ROOT)
        set(MGE_ARCH "riscv64")
    elseif(NOT "${ARM_CROSS_BUILD_ARCH}" STREQUAL "")
        set(MGE_ARCH ${ARM_CROSS_BUILD_ARCH})
    else()
        message(FATAL_ERROR "Unknown cross-compiling settings.")
    endif()
    message(STATUS "CONFIG MGE_ARCH TO ${MGE_ARCH}")
endif()

if(${MGE_ARCH} STREQUAL "AUTO")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
        set(MGE_ARCH "x86_64")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i686")
        set(MGE_ARCH "i386")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
        set(MGE_ARCH "aarch64")
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm")
        set(MGE_ARCH "armv7")
    else()
        message(FATAL_ERROR "Unknown machine architecture for MegEngine.")
    endif()
endif()

if((${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386" OR ${MGE_ARCH} STREQUAL "armv7" OR ${MGE_ARCH} STREQUAL "aarch64") AND NOT APPLE)
    option(MGE_ENABLE_CPUINFO "Build cpuinfo library for check runtime." ON)
    if(MGE_ENABLE_CPUINFO)
        message(STATUS "Enable cpuinfo runtime check and little kernel optimize.")
        add_definitions(-DMGB_ENABLE_CPUINFO_CHECK)
        include(cmake/cpuinfo.cmake)
    endif()
endif()

if(MSVC OR WIN32)
    # for cmake after 3.15.2
    cmake_policy(SET CMP0091 NEW)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

    add_compile_definitions(NOMINMAX=1 _USE_MATH_DEFINES=1 WIN32=1)
    message(STATUS "into windows build...")
    message(VERBOSE "CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
    if (${CMAKE_C_COMPILER_ID} STREQUAL "Clang-cl")
        message(FATAL_ERROR "only support clang-cl for windows build, pls check detail: scripts/cmake-build/BUILD_README.md")
    endif()
    # add flags for enable sse instruction optimize for X86, enable avx header to compile avx code
    set(WIN_FLAGS "-msse4.2 -O2 -D_AVX_ -D_AVX2_ -D__AVX__ -D__AVX2__ -D__FMA__")
    # if u CPU is cascadelake series, u can enable for performance
    # set(WIN_FLAGS "{WIN_FLAGS} -march=cascadelake -mtune=cascadelake")
    # set(WIN_FLAGS "{WIN_FLAGS} -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni")

    # for windows build
    set(WIN_FLAGS "${WIN_FLAGS} -Wno-error=implicit-int-conversion -Wno-error=double-promotion")
    set(WIN_FLAGS "${WIN_FLAGS} -Wno-error=zero-as-null-pointer-constant -Wno-error=implicit-int-conversion")
    set(WIN_FLAGS "${WIN_FLAGS} -Wno-error=float-conversion -Wno-error=shadow-field -Wno-error=covered-switch-default")
    set(WIN_FLAGS "${WIN_FLAGS} -Wno-error=deprecated  -Wno-error=documentation  -Wno-error=unreachable-code-break")
    set(WIN_FLAGS "${WIN_FLAGS} /DWIN32 -Wno-macro-redefined /D_WIN32_WINNT=0x0601 /wd4819")
    set(WIN_FLAGS "${WIN_FLAGS} /D_CRT_SECURE_NO_DEPRECATE /D_CRT_SECURE_NO_WARNINGS /DNOGDI /D_USE_MATH_DEFINES /bigobj")
    set(WIN_FLAGS "${WIN_FLAGS} /Zm500 /EHs /wd4351 /wd4291 /wd4250 /wd4996 /wd4819 -Wno-inconsistent-dllimport")

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WIN_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WIN_FLAGS}")

    #FIXME: fix halide JIT on windows
    message(STATUS "disable jit, halide and mlir on windows host build...")
    set(MGE_WITH_HALIDE OFF)
    set(MGE_WITH_JIT OFF)
    set(MGE_WITH_JIT_MLIR OFF)
    #FIXME: fix MegRay on windows
    message(STATUS "Disable distributed build on windows host build...")
    set(MGE_WITH_DISTRIBUTED OFF)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    if(ANDROID)
        set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Ofast -DNDEBUG -g")

    else()
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g")
    endif()
endif()

CHECK_CXX_COMPILER_FLAG(-fuse-ld=gold CXX_SUPPORT_GOLD)
if(CXX_SUPPORT_GOLD AND NOT ANDROID AND NOT APPLE AND NOT MSVC AND NOT WIN32)
    message(STATUS "Using GNU gold linker.")
    set(MGE_COMMON_LINKER_FLAGS "-fuse-ld=gold")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")
endif()

if(NOT MGE_WITH_JIT)
    if(MGE_WITH_HALIDE)
        message(WARNING "MGE_WITH_HALIDE is set to OFF with MGE_WITH_JIT disabled")
        set(MGE_WITH_HALIDE OFF)
    endif()
    if(MGE_WITH_JIT_MLIR)
        message(WARNING "MGE_WITH_JIT_MLIR is set to OFF with MGE_WITH_JIT disabled")
        set(MGE_WITH_JIT_MLIR OFF)
    endif()
endif()

# FIXME At present, there are some conflicts between the LLVM that halide
# depends on and the LLVM that MLIR depends on. Should be fixed in subsequent
# versions.
if(MGE_BUILD_IMPERATIVE_RT)
    set(MGE_WITH_HALIDE OFF)
    message(WARNING "cannot use HALIDE when building IMPERATIVE_RT")
endif()
if(MGE_WITH_JIT_MLIR)
    if(MGE_WITH_HALIDE)
        message(FATAL_ERROR "please set MGE_WITH_HALIDE to OFF with MGE_WITH_JIT_MLIR enabled")
    endif()
endif()

if(MGE_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found in PATH")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT MGE_WITH_CUDA)
    if(NOT MGE_ARCH STREQUAL "x86_64" AND NOT MGE_ARCH STREQUAL "i386")
        message(STATUS "Disable JIT support, as the MGE_ARCH is not X86 and CUDA is not enabled.")
        set(MGE_WITH_JIT OFF)
        set(MGE_WITH_JIT_MLIR OFF)
    endif()
    set(MGE_WITH_HALIDE OFF)
    message(STATUS "Disable TensorRT support, as CUDA is not enabled.")
    set(MGE_WITH_TRT OFF)
endif()

find_package(PythonInterp 3 REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(NOT "${CMAKE_THREAD_LIBS_INIT}" STREQUAL "")
    if(${CMAKE_THREAD_LIBS_INIT} STREQUAL "-pthread" AND MGE_WITH_CUDA)
        set_property(TARGET Threads::Threads
            PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
    endif()
endif()

set(MGE_BLAS MKL CACHE STRING "BLAS implementaion used by MegEngine.")
set_property(CACHE MGE_BLAS PROPERTY STRINGS MKL OpenBLAS)
set(MGE_CUDA_GENCODE "" CACHE STRING "Overwrite -gencode specifications for CUDA")
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER $(CMAKE_CXX_COMPILER))
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT MGE_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

if(NOT MGE_ENABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
endif()

if(MGE_WITH_TEST)
    include(cmake/gtest.cmake)
endif()

if(MGE_BUILD_IMPERATIVE_RT)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT MGE_WITH_CUDA)
    message(STATUS "Disable distributed support, as CUDA is not enabled.")
    set(MGE_WITH_DISTRIBUTED OFF)
endif()

if(MGE_INFERENCE_ONLY)
    message(STATUS "Disable distributed support for inference only build.")
    set(MGE_WITH_DISTRIBUTED OFF)
    message(STATUS "Disable imperative_rt python module for inference only build.")
    set(MGE_BUILD_IMPERATIVE_RT OFF)
endif()

if(MGE_WITH_JIT_MLIR OR MGE_BUILD_IMPERATIVE_RT)
    include(cmake/llvm-project.cmake)
endif()

if(MGE_WITH_DISTRIBUTED)
    include(cmake/protobuf.cmake)
    include(cmake/zmq.cmake)
endif()

if(MGB_WITH_FLATBUFFERS)
    include(cmake/flatbuffers.cmake)
endif()

if(MGE_WITH_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    foreach(path ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
        get_filename_component(_NAME ${path} NAME)
        if(NOT ${_NAME} STREQUAL "stubs")
            list(APPEND CUDA_LINK_DIRECTORIES ${path})
        endif()
    endforeach()
    link_directories(${CUDA_LINK_DIRECTORIES})

    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
    if(MSVC OR WIN32)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")
        set(CCBIN_FLAG "${CCBIN_FLAG} /wd4819 /wd4334 /wd4267 /wd4002 /wd4244 /wd4068")
        if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
            set(CCBIN_FLAG "${CCBIN_FLAG} -D_ITERATOR_DEBUG_LEVEL=2 -MTd")
        endif()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \" ${CCBIN_FLAG} \" ")
    else()
        set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall,-Wextra -Xfatbin -compress-all")
    endif()

    if(NOT MGE_ENABLE_RTTI)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-rtti")
    endif()
    if(NOT MGE_ENABLE_EXCEPTIONS)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-exceptions")
    endif()

    if(NOT MGE_CUDA_GENCODE)
        if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
            set(MEGDNN_THREADS_512 0)
            if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=compute_75")
            elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "9.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "9.0.0")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=compute_70")
            else()
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_35,code=sm_35")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=compute_61")
            endif()
        else()
            message(FATAL_ERROR "Unsupported CUDA host arch.")
        endif()
    else()
        set(MEGDNN_THREADS_512 1)
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MGE_CUDA_GENCODE}")
    include(cmake/cudnn.cmake)
    if(MGE_WITH_TRT)
        include(cmake/tensorrt.cmake)
    endif()
    if(MGE_CUDA_USE_STATIC)
        if(MGE_WITH_TRT)
            if(MSVC OR WIN32)
                list(APPEND MGE_CUDA_LIBS ${TRT_LIBRARY} ${CUDNN_LIBRARY})
                message(STATUS "windows TRT_LIBRARY: ${TRT_LIBRARY}")
                message(STATUS "windows CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
            else()
                list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libnvinfer libcudnn -Wl,--no-whole-archive)
            endif()
        else()
            list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libcudnn -Wl,--no-whole-archive)
        endif()
        if(MSVC OR WIN32)
            list(APPEND MGE_CUDA_LIBS cusolver.lib cublas.lib curand.lib cudart_static.lib cusparse.lib)
        else()
            list(APPEND MGE_CUDA_LIBS cusolver_static cublas_static curand_static culibos cudart_static cusparse_static)
        endif()
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
            if(MSVC OR WIN32)
                list(APPEND MGE_CUDA_LIBS cublasLt.lib)
            else()
                list(APPEND MGE_CUDA_LIBS cublasLt_static)
            endif()
        endif()
        if((${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0") AND NOT MSVC AND NOT WIN32)
            # mark all symbols from liblapack_static.a as weak to avoid
            # duplicated definition with mkl
            find_library(
                LAPACK_STATIC_PATH lapack_static
                HINTS ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
            if(NOT LAPACK_STATIC_PATH)
                message(FATAL_ERROR "liblapack_static.a not found")
            endif()
            set(LAPACK_STATIC_COPY_PATH ${CMAKE_CURRENT_BINARY_DIR}/liblapack_static_copy.a)

            # add a target that run objcopy
            add_custom_command(
                OUTPUT ${LAPACK_STATIC_COPY_PATH}
                COMMAND ${CMAKE_OBJCOPY} -w -W* ${LAPACK_STATIC_PATH} ${LAPACK_STATIC_COPY_PATH}
                VERBATIM)
            add_custom_target(lapack_static_weak_target DEPENDS ${LAPACK_STATIC_COPY_PATH})

            # create a library named "lapack_static_weak"
            add_library(lapack_static_weak STATIC IMPORTED GLOBAL)
            add_dependencies(lapack_static_weak lapack_static_weak_target)
            set_target_properties(
                lapack_static_weak PROPERTIES
                IMPORTED_LOCATION ${LAPACK_STATIC_COPY_PATH})
            list(APPEND MGE_CUDA_LIBS lapack_static_weak ${LAPACK_STATIC_COPY_PATH})
        endif()
    else()
        if(MGE_WITH_TRT)
            list(APPEND MGE_CUDA_LIBS libnvinfer)
        endif()
        list(APPEND MGE_CUDA_LIBS libcudnn)
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
            list(APPEND MGE_CUDA_LIBS cublasLt cusolver cublas curand)
        endif()
    endif()

    add_subdirectory(dnn/cuda-stub)
    if(MSVC OR WIN32)
        list(APPEND MGE_CUDA_LIBS nvrtc.lib cuda-stub)
    else()
        list(APPEND MGE_CUDA_LIBS nvrtc cuda-stub nvToolsExt)
    endif()
    set(MGE_CUDA_LIBS "${MGE_CUDA_LIBS}")
endif()

if(MGE_WITH_CAMBRICON)
    include_directories("$ENV{NEUWARE_HOME}/include")
    link_directories("$ENV{NEUWARE_HOME}/lib64")
    include(cmake/FindBANG/FindBANG.cmake)
    if (${MGE_MLU_ARCH} STREQUAL "MLU100")
        set(BANG_ARCH "100")
    elseif (${MGE_MLU_ARCH} STREQUAL "MLU1h8")
        set(BANG_ARCH "110")
    elseif (${MGE_MLU_ARCH} STREQUAL "MLU220")
        set(BANG_ARCH "220")
    elseif (${MGE_MLU_ARCH} STREQUAL "MLU270")
        set(BANG_ARCH "270")
    elseif (${MGE_MLU_ARCH} STREQUAL "MLU290")
        set(BANG_ARCH "290")
    elseif (${MGE_MLU_ARCH} STREQUAL "MLU200")
        set(BANG_ARCH "200")
    else()
        message (FATAL_ERROR "Unsupported MLU arch.")
    endif()
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=${MGE_MLU_ARCH}")
    set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -std=c++11 -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__BANG_ARCH__=${BANG_ARCH}")
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g -O0")
    elseif (${CMAKE_BUILD_TYPE} STREQUAL "Release")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
    elseif (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g -O3")
    elseif (${CMAKE_BUILD_TYPE} STREQUAL "MinSizeRel")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Os")
    endif()
    include(cmake/cnrt.cmake)
    include(cmake/cndev.cmake)
    include(cmake/cnml.cmake)
    list(APPEND MGE_CAMBRICON_LIBS libcnrt libcndev libcnml)
    set(MGE_CAMBRICON_LIBS "${MGE_CAMBRICON_LIBS}")
endif()

if (MGE_WITH_ROCM)
    include(cmake/rocm.cmake)
endif ()


if(MGE_WITH_ATLAS)
    include(cmake/aclrt.cmake)
    list(APPEND MGE_ATLAS_LIBS libascendcl)
    set(MGE_ATLAS_LIBS "${MGE_ATLAS_LIBS}")
    set(MGB_ATLAS ${MGE_WITH_ATLAS})
endif()


find_program(CCACHE_BIN ccache)
if(CCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_BIN})
    if(MGE_WITH_CUDA AND NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0")
        message(STATUS "Using ccache as CMAKE_CUDA_COMPILER_LAUNCHER")
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_BIN})
    endif()
endif()

if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
    if(${MGE_BLAS} STREQUAL "MKL")
        include(cmake/mkl.cmake)
        set(MGE_BLAS_LIBS libmkl)
    elseif(${MGE_BLAS} STREQUAL "OpenBLAS")
        include(cmake/OpenBLAS.cmake)
        set(MGE_BLAS_LIBS libopenblas)
    else()
        message(FATAL_ERROR "Unknown BLAS implementation ${MGE_BLAS}")
    endif()
endif()

# MKLDNN build
if(MGE_WITH_MKLDNN AND ${MGE_ARCH} STREQUAL "x86_64")
    include(cmake/MKL_DNN.cmake)
    set(MEGDNN_X86_WITH_MKL_DNN 1)
endif()

# RTTI
if(MGE_ENABLE_RTTI)
    set(MEGDNN_ENABLE_MANGLING 0)
    set(MEGDNN_ENABLE_RTTI 1)
else()
    set(MEGDNN_ENABLE_MANGLING 1)
    set(MEGDNN_ENABLE_RTTI 0)
endif()
set(MGB_VERBOSE_TYPEINFO_NAME ${MGE_ENABLE_RTTI})

# Logging
set(MGB_ENABLE_LOGGING ${MGE_ENABLE_LOGGING})
set(MEGDNN_ENABLE_LOGGING ${MGE_ENABLE_LOGGING})
set(MGB_ENABLE_JSON ${MGE_ENABLE_LOGGING})

# Exception
if(NOT MGE_ENABLE_EXCEPTIONS)
    message(STATUS "Exceptions disabled; MegEngine would kill itself when it is supposed to throw an exception.")
endif()
set(MGB_ENABLE_EXCEPTION ${MGE_ENABLE_EXCEPTIONS})
set(MEGDNN_ENABLE_EXCEPTIONS ${MGE_ENABLE_EXCEPTIONS})

# JIT
if(MGE_WITH_JIT AND MGE_WITH_HALIDE)
    set(HALIDE_SHARED_LIBRARY OFF CACHE BOOL "Build as a shared library")
    include(cmake/Halide.cmake)
endif()

# Thread
IF(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    message(STATUS "disable jit, halide and mlir on macos host build...")
    set(MGE_WITH_HALIDE OFF)
    set(MGE_WITH_JIT OFF)
    set(MGE_WITH_JIT_MLIR OFF)
ENDIF()

set(MGB_JIT ${MGE_WITH_JIT})
set(MGB_JIT_MLIR ${MGE_WITH_JIT_MLIR})
set(MGB_JIT_HALIDE ${MGE_WITH_HALIDE})

if(MSVC OR WIN32)
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 1)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

if(CMAKE_THREAD_LIBS_INIT OR CMAKE_USE_WIN32_THREADS_INIT)
    set(MGB_HAVE_THREAD 1)
endif()

if(MGE_WITH_TEST)
    # use intra-op multi threads
    set(MEGDNN_ENABLE_MULTI_THREADS 1)
endif()

# CUDA
set(MGB_CUDA ${MGE_WITH_CUDA})
set(MEGDNN_WITH_CUDA ${MGE_WITH_CUDA})


#ROCM
set(MGB_ROCM ${MGE_WITH_ROCM})
set(MEGDNN_WITH_ROCM ${MGE_WITH_ROCM})


# CAMBRICON
set(MGB_CAMBRICON ${MGE_WITH_CAMBRICON})
set(MEGDNN_WITH_CAMBRICON ${MGE_WITH_CAMBRICON})


# Debug info
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    set(MGB_ASSERT_LOC 1)
    set(MGB_ENABLE_DEBUG_UTIL 1)
else()
    set(MGB_ASSERT_LOC 0)
    set(MGB_ENABLE_DEBUG_UTIL 0)
endif()

# TensorRT
set(MGB_ENABLE_TENSOR_RT ${MGE_WITH_TRT})

# Inference only
if(MGE_INFERENCE_ONLY AND NOT MGE_WITH_TEST)
    set(MGB_ENABLE_GRAD 0)
    set(MGB_BUILD_SLIM_SERVING 1)
else()
    set(MGB_ENABLE_GRAD 1)
    set(MGB_BUILD_SLIM_SERVING 0)
endif()

# Distributed communication
set(MGB_ENABLE_OPR_MM ${MGE_WITH_DISTRIBUTED})

# MGE_ARCH related flags
if(MGE_ARCH STREQUAL "x86_64" OR MGE_ARCH STREQUAL "i386")
    if(MGE_BLAS STREQUAL "MKL")
        set(MEGDNN_X86_WITH_MKL 1)
    elseif(MGE_BLAS STREQUAL "OpenBLAS")
        set(MEGDNN_X86_WITH_OPENBLAS 1)
    endif()
endif()

# Enable Naive
if(MGE_ARCH STREQUAL "naive")
    set(MEGDNN_NAIVE 1)
    message(STATUS "MEGDNN_NAIVE is enabled; MegDNN performance is degraded.")
endif()

if(MGE_ARCH STREQUAL "x86_64" OR MGE_ARCH STREQUAL "i386")
    set(MEGDNN_X86 1)
    if(MGE_ARCH STREQUAL "x86_64")
        set(MEGDNN_X86_64 1)
        set(MEGDNN_64_BIT 1)
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
        endif()
    else()
        set(MEGDNN_X86_32 1)
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
        endif()
    endif()
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mfpmath=sse")
    endif()
endif()

if(MGE_ARCH STREQUAL "armv7")
    # -funsafe-math-optimizations to enable neon auto-vectorization (since neon is not fully IEEE 754 compatible, GCC does not turn on neon auto-vectorization by default.
    if(ANDROID)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=softfp -mfpu=neon")
    endif()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations")
    set (MARCH "-march=armv7-a")
    set (MEGDNN_ARMV7 1)
endif()

if(MGE_ARCH STREQUAL "aarch64")
    set(MEGDNN_AARCH64 1)
    set(MEGDNN_64_BIT 1)
    set(MARCH "-march=armv8-a")
    set(MGB_AARCH64 1)
    if(MGE_ARMV8_2_FEATURE_FP16)
        message(STATUS "Enable fp16 feature support in armv8.2")
        if(NOT ${MGE_DISABLE_FLOAT16})
            set(MEGDNN_ENABLE_FP16_NEON 1)
        endif()
        set(MARCH "-march=armv8.2-a+fp16")
    endif()

    if(MGE_ARMV8_2_FEATURE_DOTPROD)
        message(STATUS "Enable dotprod feature support in armv8.2")
        if(MGE_ARMV8_2_FEATURE_FP16)
            set(MARCH "-march=armv8.2-a+fp16+dotprod")
        else()
            set(MARCH "-march=armv8.2-a+dotprod")
        endif()
    endif()

endif()

if(MGE_ARCH STREQUAL "riscv64")
    set(MEGDNN_RISCV64 1)
    set(MEGDNN_64_BIT 1)
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH}")

set(MGE_VERSION_SCRIPT ${PROJECT_SOURCE_DIR}/src/version.ld CACHE INTERNAL "Path to linker version script")

# Write out megbrain_build_config.h
# It defines macros needed by both megbrain and dnn
configure_file(src/megbrain_build_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(dnn)

list(APPEND MGB_OPR_PARAM_DEFS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py)
set(MGB_OPR_PARAM_DEFS_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/dnn/scripts/gen_param_defs.py)

set(MGB_OPR_PARAM_DEFS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/opr/include/)
file(MAKE_DIRECTORY ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr)
add_custom_command(
    OUTPUT
        ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
    COMMAND ${PYTHON_EXECUTABLE} ${MGB_OPR_PARAM_DEFS_SCRIPT} ${MGB_OPR_PARAM_DEFS_SRCS}
        ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
    DEPENDS ${MGB_OPR_PARAM_DEFS_SRCS} ${MGB_OPR_PARAM_DEFS_SCRIPT}
    VERBATIM
)

list(APPEND MGB_OPR_PARAM_DEFS_OUTS
    ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
)

install(FILES ${MGB_OPR_PARAM_DEFS_OUTS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/megbrain/opr/)

list(APPEND MGB_OPR_PARAM_DEFS_INC ${MGB_OPR_PARAM_DEFS_OUT_DIR})
add_custom_target(_mgb_opr_param_defs DEPENDS ${MGB_OPR_PARAM_DEFS_OUTS})
add_library(mgb_opr_param_defs INTERFACE)
target_include_directories(mgb_opr_param_defs
    INTERFACE
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${MGB_OPR_PARAM_DEFS_INC}>
)
add_dependencies(mgb_opr_param_defs _mgb_opr_param_defs)
install(TARGETS mgb_opr_param_defs EXPORT ${MGE_EXPORT_TARGETS})

if(MGE_WITH_JIT_MLIR OR MGE_BUILD_IMPERATIVE_RT)
    # generate param_defs.td
    set(MGE_GENFILE_DIR ${PROJECT_BINARY_DIR}/src/genfiles)
    set(MGE_GEN_IR_DIR ${PROJECT_BINARY_DIR}/src/core/include/megbrain/ir)
    set(OPR_PARAM_DEFS_SRCS ${MGE_GENFILE_DIR}/opr_param_defs.py)
    set(OPR_PARAM_DEFS_SCRIPT ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_tablegen.py)
    set(OPR_PARAM_DEFS_OUT ${MGE_GEN_IR_DIR}/param_defs.td)
    file(COPY ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py DESTINATION ${MGE_GENFILE_DIR})
    file(READ ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py CONTENTS)
    file(APPEND ${OPR_PARAM_DEFS_SRCS} ${CONTENTS})
    file(MAKE_DIRECTORY ${MGE_GEN_IR_DIR})
    add_custom_target(param_defs_tblgen
        COMMAND ${PYTHON_EXECUTABLE} ${OPR_PARAM_DEFS_SCRIPT} ${OPR_PARAM_DEFS_SRCS} ${OPR_PARAM_DEFS_OUT}
        DEPENDS ${OPR_PARAM_DEFS_SRCS} ${OPR_PARAM_DEFS_SCRIPT}
        VERBATIM
    )
    # mlir tblgen sources
    set(MGE_IR_DIR ${PROJECT_SOURCE_DIR}/src/core/include/megbrain/ir)
    set(MGE_IR_INCLUDE_DIRS ${MLIR_LLVM_INCLUDE_DIR} ${MGE_IR_DIR} ${MGE_GEN_IR_DIR})
    list(TRANSFORM MGE_IR_INCLUDE_DIRS PREPEND "-I")
    file(GLOB_RECURSE MGE_IR_TDS ${MGE_IR_DIR}/*.td)
endif()

if(MGE_WITH_DISTRIBUTED)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/MegRay)
endif()

add_subdirectory(src)

if(MGE_BUILD_SDK)
    add_subdirectory(sdk/load-and-run)
endif()


if(MGE_BUILD_IMPERATIVE_RT)
    add_subdirectory(imperative)
    message(STATUS "Enable imperative python wrapper runtime")
endif()

if(MGE_WITH_TEST AND MGE_ENABLE_RTTI)
    add_subdirectory(test)
endif()

if(TARGET _imperative_rt)
    add_custom_target(
        develop
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/imperative/python/${PACKAGE_NAME}/core/$<TARGET_FILE_NAME:${MODULE_NAME}>
          ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/core/$<TARGET_FILE_NAME:${MODULE_NAME}>
        DEPENDS _imperative_rt
        VERBATIM
    )
endif()

# Configure and install pkg-config.
# Note that unlike the Config.cmake modules, this is not relocatable (and not
# really portable) because we have two dependencies without pkg-config
# descriptions: FlatBuffers and MKL-DNN
if (MGE_USE_SYSTEM_MKLDNN)
    set (MGE_PKGCONFIG_LIBS_PRIVATE "-ldnnl")
endif()
if (MGE_USE_SYSTEM_OPENBLAS)
    set (MGE_PKGCONFIG_LIBS_PRIVATE "${MGE_PKGCONFIG_LIBS_PRIVATE} -lopenblas")
endif()
configure_file(cmake/megengine.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/megengine.pc
               @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/megengine.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Do not export targets if MGE_WITH_DISTRIBUTED is on. MegRay is not ready.
if (NOT MGE_WITH_DISTRIBUTED)
    include(CMakePackageConfigHelpers)
    set (MGE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/MegEngine)
    configure_package_config_file(cmake/MegEngineConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfig.cmake
        INSTALL_DESTINATION ${MGE_INSTALL_CMAKEDIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfigVersion.cmake
        VERSION ${MGB_VER_MAJOR}.${MGB_VER_MINOR}.${MGB_VER_PATCH}
        COMPATIBILITY SameMajorVersion)

    install(EXPORT ${MGE_EXPORT_TARGETS} DESTINATION ${MGE_INSTALL_CMAKEDIR})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfigVersion.cmake
            DESTINATION ${MGE_INSTALL_CMAKEDIR})
endif()

if(MSVC OR WIN32)
    add_compile_options(
        $<$<CONFIG:>:/MT>
        $<$<CONFIG:Debug>:/MTd>
        $<$<CONFIG:Release>:/MT>
        )
    foreach (CompilerFlag
            CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${CompilerFlag} MATCHES "/MD")
            string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
            set(${CompilerFlag} "${${CompilerFlag}}" CACHE STRING "msvc compiler flags" FORCE)
            message(VERBOSE "MSVC flags: ${CompilerFlag}:${${CompilerFlag}}")
        endif()
    endforeach()
endif()

if(MGE_WITH_JIT_MLIR)
    add_subdirectory(tools/mlir/mgb-opt)
    add_subdirectory(tools/mlir/mgb-file-check)
endif()
