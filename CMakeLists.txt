cmake_minimum_required(VERSION 3.15.2)
message(STATUS "CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
if(NOT ${CMAKE_GENERATOR} STREQUAL "Ninja")
  message(WARNING "CMAKE_GENERATOR NOT EQUAL Ninja, which we do not recommend")
endif()

include(cmake/FetchMegBrainVersion.cmake)
project(
  MegEngine
  LANGUAGES C CXX
  VERSION ${MGB_VER_STRING})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

if(NOT MSVC
   AND NOT APPLE
   AND NOT WIN32)
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>")
endif()

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)
include(CMakeDependentOption)

check_cxx_compiler_flag(-Wclass-memaccess CXX_SUPPORT_WCLASS_MEMACCESS)

set(MGE_ARCH
    AUTO
    CACHE STRING "Architecture on which MegEngine to be built.")
set_property(
  CACHE MGE_ARCH
  PROPERTY STRINGS
           AUTO
           x86_64
           i386
           armv7
           aarch64
           naive
           fallback)
set(MGE_EXPORT_TARGETS MegEngine-targets)

if(NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
  string(REPLACE ":" ";" ALTER_LD_LIBRARY_PATHS $ENV{LD_LIBRARY_PATH})
else()
  set(ALTER_LD_LIBRARY_PATHS "")
endif()

if(NOT "$ENV{LIBRARY_PATH}" STREQUAL "")
  string(REPLACE ":" ";" ALTER_LIBRARY_PATHS $ENV{LIBRARY_PATH})
else()
  set(ALTER_LIBRARY_PATHS "")
endif()

option(MGE_WITH_JIT "Build MegEngine with JIT." ON)
option(MGE_WITH_JIT_MLIR "Build MegEngine with MLIR JIT." OFF)
option(MGE_WITH_HALIDE "Build MegEngine with Halide JIT" OFF)
option(MGE_WITH_MIDOUT_PROFILE "Build MegEngine with Midout profile." OFF)
option(
  MGE_WITH_MINIMUM_SIZE
  "Swith off MGE_ENABLE_RTTI、MGE_ENABLE_EXCEPTIONS、MGE_ENABLE_LOGGING and switch on MGE_INFERENCE_ONLY so that compile minimum load_and_run."
  OFF)
option(MGE_ARMV8_2_FEATURE_FP16 "Enable armv8.2-a+fp16 support" OFF)
option(MGE_DISABLE_FLOAT16 "Disable MegEngine float16 support." OFF)
option(MGE_WITH_CUDA "Enable MegEngine CUDA support." ON)
option(MGE_CUDA_USE_STATIC "Enable MegEngine CUDA static linking." ON)
option(MGE_WITH_LITE "Build MGE with lite" ON)
option(MGE_WITH_TRT "Build MegEngine with TensorRT." ON)
option(MGE_WITH_CUDA_STUB "Build MegEngine with CUDA stub." ON)
option(MGE_WITH_NVRTC_STUB "Build MegEngine with NVRTC stub." OFF)
option(MGE_WITH_CUDNN_SHARED "Build MegEngine with CUDNN shared." ON)
option(MGE_WITH_CUBLAS_SHARED "Build MegEngine with CUBLAS shared." OFF)
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
option(MGE_WITH_BENCHMARK "Enable DNN BENCHMARK" OFF)
option(MGE_WITH_DISTRIBUTED "Build with distributed support" ON)
option(MGE_BUILD_IMPERATIVE_RT "Build _imperative_rt Python Module " ON)
option(MGE_INFERENCE_ONLY "Build inference only library." OFF)
option(MGE_WITH_MKLDNN "Enable Intel MKL_DNN support," ON)
option(MGE_WITH_ROCM "Enable ROCM support" OFF)
option(MGE_WITH_LARGE_ARCHIVE "Enable big archive link support" OFF)
option(MGE_BUILD_WITH_ASAN "Enable build with ASAN, need compiler support" OFF)
option(MGE_WITH_CUSTOM_OP "Build with Custom op" OFF)
option(MGE_SYNC_THIRD_PARTY "help sync third_party submodule" OFF)
option(MGE_PROFILE_COMPILE_TIME "help profile compile time per file" OFF)

if(MGE_PROFILE_COMPILE_TIME)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "cmake -E time")
endif()

# TODO: add windows support
cmake_dependent_option(MGE_WITH_CUPTI "Build with CUPTI" OFF
                       "MGE_WITH_CUDA;MGE_BUILD_IMPERATIVE_RT;NOT MSVC;NOT WIN32" OFF)

set(MGB_CUPTI ${MGE_WITH_CUPTI})

if(MSVC OR WIN32)
  # FIXME: static link Windows vc runtime with some version from Visual Studio have some
  # runtime issue at some call PATH, for example: _imperative_rt.pyd -->
  # megengine_shared.dll for example c api flush can not find the fd args, I have no
  # idea about this issue as a Workround, dynamic link vc runtime,  but at some case, we
  # will static link vcrt when
  # MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP/MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2, so please
  # use lite_static_all_in_one(lite/CMakeLists.txt) in Windows XP env as possible How to
  # install VC runtime if you env do not install, refer to:
  # https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-160
  option(MGE_STATIC_LINK_WITH_VC_RUNTIME
         "Enable mge static link with Windows vc runtime" OFF)

  option(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP "Enable deploy inference on Windows xp" OFF)
  # special MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2 for Windows XP sp2(32bit) internal
  # behavior: 1: will force define MGB_HAVE_THREAD=0, which means only support single
  # thread 2: some Feature will be disable, eg: MGB_ENABLE_JSON and var sanity check, do
  # not too many care this!!, if you want to use this Feature to 'DEBUG', you can run
  # same model at NON-XP-SP2 env, eg Win7 or XP-SP3(build without
  # MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2) 3: we only support MegEngine(load_and_run)
  # and MegEngineLite API work on XP SP2 some debug utils, eg, megbrain_test/megdnn_test
  # not support run, most caused by gtest src code sdk caller: 1: as we remove mutex,
  # when you use MSVC self API eg CreateThread to start several MegEngine instances in
  # the same progress, please call MegEngine API(init/run) as serial as possible, also
  # please do not use std::thread std::mutex/std::this_thread_id at SDK caller side!!!
  # check dll/exe can deploy on Windows XP sp2 or not: please checkout
  # scripts/misc/check_windows_xp_sp2_deploy.py
  option(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2
         "Enable deploy inference on Windows xp sp2" OFF)

  # PE file linked by LLVM lld can not run at Windows XP env, so we force use link.exe
  # which always locate in Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/*/*/link.exe
  set(CMAKE_LINKER "link.exe")
  if(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP OR MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
    set(MGE_STATIC_LINK_WITH_VC_RUNTIME ON)
    message(
      STATUS "Force set MGE_STATIC_LINK_WITH_VC_RUNTIME ON when build for Windows XP")

    if(NOT ${MGE_ARCH} STREQUAL "i386")
      message(FATAL_ERROR "only support 32bit when build for Windows xp")
    endif()

    if(NOT MGE_INFERENCE_ONLY)
      message(FATAL_ERROR "only support inference when build for Windows xp")
    endif()

    if(MGE_WITH_CUDA)
      message(FATAL_ERROR "do not support CUDA when build for Windows xp")
    endif()

    # Windows XP sp3 have thread issue, Workround for it
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32_WINNT=0x0501 /Zc:threadSafeInit-")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_WIN32_WINNT=0x0501 /Zc:threadSafeInit-")
    # for Windows XP type
    add_link_options("/SUBSYSTEM:CONSOLE,5.01")
    # some old lib(for example mkl for xp) use legacy stdio, so we force link
    # legacy_stdio_definitions
    add_link_options("/DEFAULTLIB:legacy_stdio_definitions.lib")

    if(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__DEPLOY_ON_XP_SP2__=1")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__DEPLOY_ON_XP_SP2__=1")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_WIN32_WINNT=0x0601")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D_WIN32_WINNT=0x0601")
  endif()
endif()

if(MSVC OR WIN32)
  message(STATUS "windows force cudnn static link")
  set(MGE_WITH_CUDNN_SHARED OFF)
endif()

if(MGE_WITH_NVRTC_STUB OR MGE_WITH_CUDA_STUB)
  set(MGE_WITH_ANY_CUDA_STUB ON)
else()
  set(MGE_WITH_ANY_CUDA_STUB OFF)
endif()

if(MGE_WITH_MIDOUT_PROFILE)
  message(
    STATUS
      "build with MIDOUT PROFILE and force set MGE_WITH_MINIMUM_SIZE off and force rtti ON"
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMIDOUT_PROFILING")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMIDOUT_PROFILING")
  set(MGE_WITH_MINIMUM_SIZE OFF)
  set(MGE_ENABLE_RTTI ON)
  if(WIN32)
    message(FATAL_ERROR "do not support midout at WIN32")
  endif()
endif()

set(BIN_REDUCE ${PROJECT_SOURCE_DIR}/src/bin_reduce_cmake.h)
if(MGE_WITH_MINIMUM_SIZE)
  message(STATUS "build with MGE_WITH_MINIMUM_SIZE bin_reduce header is: ${BIN_REDUCE}")
  set(MGE_ENABLE_RTTI OFF)
  set(MGE_ENABLE_LOGGING OFF)
  set(MGE_ENABLE_EXCEPTIONS OFF)
  set(MGE_INFERENCE_ONLY ON)
  # MGE_WITH_MINIMUM_SIZE will triger unused-parameter
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
endif()

if(NOT MGE_WITH_MIDOUT_PROFILE AND NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -include ${BIN_REDUCE}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -include ${BIN_REDUCE}")
endif()

check_cxx_compiler_flag(-fuse-ld=gold CXX_SUPPORT_GOLD)

if(NOT APPLE)
  # check CXX_FUNCTION_DATA_GC_SECTIONS_SUPPORT on APPLE will leak cmake crash
  check_cxx_compiler_flag("-ffunction-sections -fdata-sections  -Wl,--gc-sections"
                          CXX_FUNCTION_DATA_GC_SECTIONS_SUPPORT)
  if(CXX_FUNCTION_DATA_GC_SECTIONS_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    # check more -Wl,-z,nocopyreloc -Wl,--icf=all to reduce elf size -fuse-ld=gold is
    # not compat with -Wl,-z,nocopyreloc -Wl,--icf=all so we only try enable icf on
    # ANDROID
    check_cxx_compiler_flag("-Wl,-z,nocopyreloc -Wl,--icf=all -fuse-ld=lld"
                            LINKER_SUPPORT_Z_NOCOPYRELOC_ICF_ALL)
    check_cxx_compiler_flag("-Wl,-z,nocopyreloc -Wl,--icf=safe"
                            LINKER_SUPPORT_Z_NOCOPYRELOC_ICF_SAFE_NO_LLD)
    if(LINKER_SUPPORT_Z_NOCOPYRELOC_ICF_ALL AND (ANDROID OR OHOS))
      message(STATUS "icf is supported in this compiler")
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,nocopyreloc -Wl,--icf=all -fuse-ld=lld")
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,nocopyreloc -Wl,--icf=all -fuse-ld=lld")
    elseif(LINKER_SUPPORT_Z_NOCOPYRELOC_ICF_SAFE_NO_LLD)
      message(STATUS "icf=safe is supported in this compiler without lld")
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,nocopyreloc -Wl,--icf=safe")
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,nocopyreloc -Wl,--icf=safe")
      if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fPIC>)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
      endif()
    endif()

  endif()
endif()

check_ipo_supported(RESULT IS_LTO_SUPPORT OUTPUT output_info)
# LLVM on Windows report support LTO, but do not support -flto=full at link stage
if(IS_LTO_SUPPORT AND NOT WIN32)
  message(STATUS "lto is supported in this compiler")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto=full")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto=full")
else()
  message(STATUS "lto is not supported in this compiler")
endif()

if(APPLE)
  set(BUILD_SHARED_LIBS OFF)
  message(STATUS "build static for xcode framework require")
endif()

if(MGE_USE_SYSTEM_LIB)
  set(MGE_CUDA_USE_STATIC OFF)
endif()

if(MGB_WITH_FLATBUFFERS)
  set(MGB_ENABLE_FBS_SERIALIZATION ON)
endif()

if(OHOS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
endif()

if(CMAKE_TOOLCHAIN_FILE)
  message(STATUS "We are cross compiling.")
  message(
    STATUS
      "config FLATBUFFERS_FLATC_EXECUTABLE to: ${PROJECT_SOURCE_DIR}/build_dir/host_flatc/install/bin/flatc"
  )
  set(FLATBUFFERS_FLATC_EXECUTABLE
      "${PROJECT_SOURCE_DIR}/build_dir/host_flatc/install/bin/flatc")
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
  elseif(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
    if(${OHOS_ARCH} STREQUAL "armeabi-v7a")
      set(MGE_ARCH "armv7")
    elseif(${OHOS_ARCH} STREQUAL "arm64-v8a")
      set(MGE_ARCH "aarch64")
    else()
      message(FATAL_ERROR "DO NOT SUPPORT OHOS ARCH NOW")
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
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL
                                                    "AMD64")
    set(MGE_ARCH "x86_64")
  elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386" OR ${CMAKE_SYSTEM_PROCESSOR}
                                                      STREQUAL "i686")
    set(MGE_ARCH "i386")
  elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64" OR ${CMAKE_SYSTEM_PROCESSOR}
                                                         STREQUAL "arm64")
    set(MGE_ARCH "aarch64")
  elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^arm")
    set(MGE_ARCH "armv7")
  else()
    message(FATAL_ERROR "Unknown machine architecture for MegEngine.")
  endif()
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release"
   AND NOT MGE_WITH_TEST
   AND NOT ${MGE_ARCH} STREQUAL "x86_64"
   AND NOT MGE_WITH_MIDOUT_PROFILE)
  set(MGE_ENABLE_RTTI OFF)
  message(
    STATUS
      "disable MGE_ENABLE_RTTI when Release/NON-x86_64/NON-MGE_WITH_MIDOUT_PROFILE mode!!"
  )
endif()

if(MSVC OR WIN32)
  # for cmake after 3.15.2
  cmake_policy(SET CMP0091 NEW)
  set(CMAKE_OBJECT_PATH_MAX 300)
  if(MGE_BUILD_WITH_ASAN)
    set(MGE_STATIC_LINK_WITH_VC_RUNTIME ON)
    message(
      STATUS
        "Force set MGE_STATIC_LINK_WITH_VC_RUNTIME ON when build for Windows MGE_BUILD_WITH_ASAN"
    )
  endif()
  if(MGE_STATIC_LINK_WITH_VC_RUNTIME)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()
  else()
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL")
    else()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
  endif()

  add_compile_definitions(NOMINMAX=1 _USE_MATH_DEFINES=1 WIN32=1)
  message(STATUS "into windows build CMAKE_C_COMPILER_ID: ${CMAKE_C_COMPILER_ID}")
  if(NOT ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND NOT ${CMAKE_C_COMPILER_ID}
                                                     STREQUAL "Clang-cl")
    message(
      FATAL_ERROR
        "only support clang-cl for windows build, pls check detail: scripts/cmake-build/BUILD_README.md"
    )
  endif()
  # on windows need append
  # VS_PATH/VC/Tools/Llvm/x64/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/windows and
  # VS_PATH/VC/Tools/Llvm/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/windows to PATH
  # env
  if(MGE_BUILD_WITH_ASAN)
    message(
      WARNING
        "please do (set)export ASAN_OPTIONS=windows_hook_rtl_allocators=true when run test after build finish, caused by we link asan dll!!"
    )
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      message(
        WARNING
          "Windows AddressSanitizer doesn't support linking with debug runtime libraries yet, which means do not support CMAKE_BUILD_TYPE=Debug"
      )
      message(
        FATAL_ERROR
          "Please build with RelWithDebInfo or Release by : EXTRA_CMAKE_ARGS=\"-DMGE_BUILD_WITH_ASAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ...\""
      )
    endif()
    if("$ENV{VS_PATH}" STREQUAL "")
      message(
        FATAL_ERROR
          "can not find VS_PATH, please export Visual Studio root dir to VS_PATH env")
    endif()
    if(${MGE_ARCH} STREQUAL "x86_64")
      set(WINDOWS_ASAN_DLL_NAME "clang_rt.asan_dynamic-x86_64.lib")
      set(WINDOWS_ASAN_RUNTIME_THUNK_NAME "clang_rt.asan_dynamic_runtime_thunk-x86_64")
      set(WINDOWS_ASAN_PATH_SUFFIXES
          "VC/Tools/Llvm/x64/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/windows")
    elseif(${MGE_ARCH} STREQUAL "i386")
      set(WINDOWS_ASAN_DLL_NAME "clang_rt.asan_dynamic-i386.lib")
      set(WINDOWS_ASAN_RUNTIME_THUNK_NAME
          "clang_rt.asan_dynamic_runtime_thunk-i386.lib")
      set(WINDOWS_ASAN_PATH_SUFFIXES
          "VC/Tools/Llvm/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/windows")
    else()
      message(FATAL_ERROR "unsupport asan ARCH: ${MGE_ARCH} on Windows")
    endif()
    find_path(
      ASAN_DLL_PATH
      NAMES ${WINDOWS_ASAN_DLL_NAME}
      HINTS $ENV{VS_PATH}
      PATH_SUFFIXES ${WINDOWS_ASAN_PATH_SUFFIXES}
      DOC "Windows asan library path")
    if(ASAN_DLL_PATH STREQUAL "ASAN_DLL_PATH-NOTFOUND")
      message(FATAL_ERROR "can not find asan dll, please upgrade you LLVM")
    endif()

    message(STATUS "Windows asan dll path: ${ASAN_DLL_PATH}")
    link_directories(${ASAN_DLL_PATH})
    link_libraries(${WINDOWS_ASAN_DLL_NAME})
    link_libraries(${WINDOWS_ASAN_RUNTIME_THUNK_NAME})
    set(WIN_FLAGS "/Od -DNDEBUG -fsanitize=address")
    # windows Llvm asan do not take effect when /O2 RELWITHDEBINFO default value is /O2,
    # so override it
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "/Zi /Od /Ob1 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /Od /Ob1 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/Zi /Od /Ob1 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "/Zi /Od /Ob1 /DNDEBUG")
  else()
    set(WIN_FLAGS "/O2")
  endif()
  # add flags for enable sse instruction optimize for X86, enable avx header to compile
  # avx code
  set(WIN_FLAGS "${WIN_FLAGS} -msse4.2 -D_AVX_ -D_AVX2_ -D__AVX__ -D__AVX2__ -D__FMA__")
  # if u CPU is cascadelake series, u can enable for performance set(WIN_FLAGS
  # "{WIN_FLAGS} -march=cascadelake -mtune=cascadelake") set(WIN_FLAGS "{WIN_FLAGS}
  # -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni")

  # for windows build
  set(WIN_FLAGS
      "${WIN_FLAGS} -Wno-error=implicit-int-conversion -Wno-error=double-promotion")
  set(WIN_FLAGS
      "${WIN_FLAGS} -Wno-error=zero-as-null-pointer-constant -Wno-error=implicit-int-conversion"
  )
  set(WIN_FLAGS
      "${WIN_FLAGS} -Wno-error=float-conversion -Wno-error=shadow-field -Wno-error=covered-switch-default"
  )
  set(WIN_FLAGS
      "${WIN_FLAGS} -Wno-error=deprecated  -Wno-error=documentation  -Wno-error=unreachable-code-break"
  )
  set(WIN_FLAGS "${WIN_FLAGS} /DWIN32 -Wno-macro-redefined /wd4819")
  set(WIN_FLAGS
      "${WIN_FLAGS} /D_CRT_SECURE_NO_DEPRECATE /D_CRT_SECURE_NO_WARNINGS /DNOGDI /D_USE_MATH_DEFINES /bigobj"
  )
  set(WIN_FLAGS
      "${WIN_FLAGS} /Zm500 /EHs /wd4351 /wd4291 /wd4250 /wd4996 /wd4819 -Wno-inconsistent-dllimport"
  )

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WIN_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WIN_FLAGS}")

  # FIXME: fix halide/mlir JIT backends on windows
  message(STATUS "disable halide and mlir jit backends on windows host build...")
  set(MGE_WITH_HALIDE OFF)
  set(MGE_WITH_JIT_MLIR OFF)
  # TODO: imp ExecutableHelperImpl@src/jit/impl/utils.cpp build with Windows, then
  # enable base jit on Windows
  message(STATUS "disable base jit on windows host build...")
  set(MGE_WITH_JIT OFF)
  # FIXME: fix MegRay on windows
  message(STATUS "Disable distributed build on windows host build...")
  set(MGE_WITH_DISTRIBUTED OFF)
  if(${MGE_ARCH} STREQUAL "i386" AND ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    # https://docs.microsoft.com/en-us/cpp/build/reference/z7-zi-zi-debug-information-format?view=msvc-170
    # Workround for error LNK1318
    message(
      STATUS
        "force use full symbolic debugging with build for 32bit for Windows with Debug mode"
    )
    set(CMAKE_C_FLAGS_DEBUG "/Z7")
    set(CMAKE_CXX_FLAGS_DEBUG "/Z7")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

  # NONE windows DEBUG general flags
  if(MGE_BUILD_WITH_ASAN)
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address -fno-omit-frame-pointer")
  else()
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
  endif()

  # NONE windows opt general flags
  if(MGE_BUILD_WITH_ASAN)
    set(OPTIMIZE_LEVEL "-g -O0 -DNDEBUG -fsanitize=address -fno-omit-frame-pointer")
  elseif(ANDROID)
    set(OPTIMIZE_LEVEL "-g -Ofast -DNDEBUG")
  elseif(OHOS)
    set(OPTIMIZE_LEVEL "-g -Ofast -DNDEBUG")
  else()
    set(OPTIMIZE_LEVEL "-g -O3 -DNDEBUG")
  endif()
  # remove finite-math-only opt from Ofast, caused by clang have a different runtime
  # finite math logic, this issue do not find at g++, but as a unity build flags, we
  # force add -fno-finite-math-only when compiler support
  check_cxx_compiler_flag("-fno-finite-math-only" CXX_NO_FINITE_MATH_ONLY_SUPPORT)
  if(CXX_NO_FINITE_MATH_ONLY_SUPPORT)
    message(STATUS "force add -fno-finite-math-only for this compiler")
    set(OPTIMIZE_LEVEL "${OPTIMIZE_LEVEL} -fno-finite-math-only")
  endif()
  set(CMAKE_C_FLAGS_RELEASE "${OPTIMIZE_LEVEL}")
  set(CMAKE_CXX_FLAGS_RELEASE "${OPTIMIZE_LEVEL}")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "${OPTIMIZE_LEVEL}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${OPTIMIZE_LEVEL}")
  # some gnu(gcc) compiler use -static -libasan have runtime issue also, when target is
  # big, clang ld will take a long long long time when use -static-libsan, so we use
  # dynamic asan by default ANDROID asan.so depends on log, so broadcast log
  # link_libraries for megengine depends target, for example flatc target
  if(MGE_BUILD_WITH_ASAN)
    if(ANDROID)
      link_libraries(log)
    elseif(OHOS)
      link_libraries(hilog_ndk.z)
    endif()
  endif()
endif()

if(MGE_WITH_CUDA)
  include(cmake/cudnn.cmake)
  if(MGE_CUDA_USE_STATIC
     AND ("${CUDNN_VERSION}" VERSION_GREATER "8.0.0" OR "${CUDNN_VERSION}" VERSION_EQUAL
                                                        "8.0.0")
     AND (NOT MGE_WITH_CUDNN_SHARED))
    message(WARNING "Static link CUDNN8 will auto enable MGE_WITH_LARGE_ARCHIVE=ON")
    set(MGE_WITH_LARGE_ARCHIVE ON)
  endif()
endif()
if(MGE_WITH_LARGE_ARCHIVE)
  message(STATUS "Set -mcmodel=large and disable -fuse-ld=gold")
  set(MGE_COMMON_LINKER_FLAGS "-mcmodel=large")
elseif(
  CXX_SUPPORT_GOLD
  AND NOT ANDROID
  AND NOT OHOS
  AND NOT APPLE
  AND NOT MSVC
  AND NOT WIN32
  AND NOT MGE_WITH_LARGE_ARCHIVE
  AND NOT ${MGE_ARCH} STREQUAL "riscv64")
  message(STATUS "Using GNU gold linker.")
  set(MGE_COMMON_LINKER_FLAGS "-fuse-ld=gold")
endif()
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MGE_COMMON_LINKER_FLAGS}")

if(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
  # x86 cpu jit backends only support MLIR now, but MLIR runtime do not support at xp
  # sp2
  message(WARNING "disable MGE_WITH_JIT when build for windows xp sp2")
  set(MGE_WITH_JIT OFF)
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

# FIXME At present, there are some conflicts between the LLVM that halide depends on and
# the LLVM that MLIR depends on. Should be fixed in subsequent versions.
if(MGE_BUILD_IMPERATIVE_RT AND MGE_WITH_HALIDE)
  message(FATAL_ERROR "cannot use HALIDE when building IMPERATIVE_RT")
endif()
if(MGE_WITH_JIT_MLIR AND MGE_WITH_HALIDE)
  message(FATAL_ERROR "cannot use HALIDE with MGE_WITH_JIT_MLIR enabled")
endif()

if(MGE_WITH_CUDA)
  # FIXME: check_language(CUDA) failed when sbsa mode! detail:
  # https://gitlab.kitware.com/cmake/cmake/-/issues/20676
  if(CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    message(
      WARNING
        "force set CMAKE_CUDA_HOST_COMPILER to CMAKE_CXX_COMPILER when nvcc sbsa mode!!"
    )
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER AND NOT CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "CUDA compiler not found in PATH")
  endif()

  # remove this after CMAKE fix nvcc sbsa
  if(NOT CMAKE_CUDA_COMPILER AND CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_CUDA_COMPILER "nvcc")
    message(WARNING "force set CMAKE_CUDA_COMPILER to nvcc when nvcc sbsa mode!!")
  endif()

  find_package(CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(CMAKE_TOOLCHAIN_FILE)
  # TODO: fix cross build mlir-linalg-ods-gen for enable cross build with MLIR
  message(
    STATUS
      "Disable MLIR jit backends support, as we do not support cross build MLIR module caused by mlir-linalg-ods-gen, if you really need this, try build at host env, for example Android termux env for android, arm-linux env for arm with linux board"
  )
  set(MGE_WITH_JIT_MLIR OFF)
endif()

if(NOT MGE_WITH_CUDA)
  message(STATUS "Disable TensorRT support and disable HALIDE, as CUDA is not enabled.")
  set(MGE_WITH_HALIDE OFF)
  set(MGE_WITH_TRT OFF)
endif()

find_package(PythonInterp 3 REQUIRED)
# NOTICE: just use for target, which do not depend on python api PURPOSE: reuse target
# obj when switch python3 version will fallback to PYTHON_EXECUTABLE if can not find in
# PATH env
set(PYTHON3_IN_ENV "python3")
find_program(PYTHON3_EXECUTABLE_WITHOUT_VERSION ${PYTHON3_IN_ENV})
if(PYTHON3_EXECUTABLE_WITHOUT_VERSION)
  message(STATUS "use ${PYTHON3_IN_ENV} as PYTHON3_EXECUTABLE_WITHOUT_VERSION")
  set(PYTHON3_EXECUTABLE_WITHOUT_VERSION ${PYTHON3_IN_ENV})
else()
  message(
    STATUS
      "fallback ${PYTHON_EXECUTABLE} as PYTHON3_EXECUTABLE_WITHOUT_VERSION,\
    target which depend on PYTHON3_EXECUTABLE_WITHOUT_VERSION will be rebuild when switch python3"
  )
  set(PYTHON3_EXECUTABLE_WITHOUT_VERSION ${PYTHON_EXECUTABLE})
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(NOT "${CMAKE_THREAD_LIBS_INIT}" STREQUAL "")
  if(${CMAKE_THREAD_LIBS_INIT} STREQUAL "-pthread" AND MGE_WITH_CUDA)
    set_property(
      TARGET Threads::Threads
      PROPERTY INTERFACE_COMPILE_OPTIONS
               "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>"
               "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
  endif()
endif()

set(MGE_BLAS
    MKL
    CACHE STRING "BLAS implementaion used by MegEngine.")
set_property(CACHE MGE_BLAS PROPERTY STRINGS MKL OpenBLAS)
set(MGE_CUDA_GENCODE
    ""
    CACHE STRING "Overwrite -gencode specifications for CUDA")
if(NOT CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER $(CMAKE_CXX_COMPILER))
endif()

if(NOT MGE_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

if(NOT MGE_ENABLE_EXCEPTIONS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  # some lite macro will triger format-security error when disable exceptions
  add_definitions(-Wno-format-security)
endif()

if(MGE_BUILD_IMPERATIVE_RT
   OR ANDROID
   OR OHOS)
  message(STATUS "config cxx standard to 17.")
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT ${MGE_WITH_CUDA}
   AND NOT ${MGE_WITH_ROCM}
   AND NOT ${MGE_WITH_CAMBRICON}
   AND NOT ${MGE_WITH_ATLAS})
  message(
    STATUS
      "Disable distributed support, as CUDA, ROCm, CAMBRICON and ATLAS are disabled.")
  set(MGE_WITH_DISTRIBUTED OFF)
endif()

if(MGE_INFERENCE_ONLY)
  message(STATUS "Disable distributed support for inference only build.")
  set(MGE_WITH_DISTRIBUTED OFF)
  message(STATUS "Disable imperative_rt python module for inference only build.")
  set(MGE_BUILD_IMPERATIVE_RT OFF)
endif()

# please do any include(cmake/* after do this execute_process
if(MGE_SYNC_THIRD_PARTY)
  include(cmake/third_party_sync.cmake)
endif()

if(MGE_WITH_TEST)
  include(cmake/gtest.cmake)
endif()

include(cmake/gflags.cmake)

if(MGE_WITH_JIT_MLIR OR MGE_BUILD_IMPERATIVE_RT)
  include(cmake/llvm-project.cmake)
endif()

if(MGE_BUILD_IMPERATIVE_RT)
  set(MGE_WITH_CUSTOM_OP ON)
endif()

if(MGE_WITH_DISTRIBUTED)
  include(cmake/protobuf.cmake)
  include(cmake/zmq.cmake)
endif()

if(MGB_WITH_FLATBUFFERS)
  include(cmake/flatbuffers.cmake)
endif()

if(MGE_WITH_CUPTI)
  include(cmake/cupti.cmake)
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
    set(CCBIN_FLAG
        "${CCBIN_FLAG} /wd4819 /wd4334 /wd4267 /wd4002 /wd4244 /wd4068 /std:c++14 /bigobj"
    )
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
    if(${MGE_ARCH} STREQUAL "x86_64"
       OR ${MGE_ARCH} STREQUAL "i386"
       OR ${MGE_ARCH} STREQUAL "aarch64")
      set(MEGDNN_THREADS_512 0)
      # ON windows platform, static library just a shell, always fallback to DLL
      if(MGE_WITH_CUDA
         AND MGE_CUDA_USE_STATIC
         AND NOT MSVC
         AND NOT WIN32
         AND ("${CUDNN_VERSION}" VERSION_GREATER "8.0.0" OR "${CUDNN_VERSION}"
                                                            VERSION_EQUAL "8.0.0")
         AND (NOT MGE_WITH_CUDNN_SHARED))
        message(
          WARNING
            "Static link CUDNN8 with many sm is unworkable, we only enable sm61 sm70 sm75 by default, and enable MGE_WITH_LARGE_ARCHIVE=ON"
        )
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "11.8.0"
             OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "11.8.0")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_80,code=sm_80")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_86,code=sm_86")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_89,code=sm_89")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_90,code=sm_90")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_90,code=compute_90")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "11.1.0"
             OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "11.1.0")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_80,code=sm_80")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_86,code=sm_86")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_86,code=compute_86")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "11.0.0"
             OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "11.0.0")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_80,code=sm_80")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_80,code=compute_80")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0"
             OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=compute_75")
      elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "9.0.0"
             OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "9.0.0")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=compute_70")
      else()
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_35,code=sm_35")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MGE_CUDA_GENCODE
            "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=compute_61")
      endif()
    else()
      message(FATAL_ERROR "Unsupported CUDA host arch.")
    endif()
  else()
    set(MEGDNN_THREADS_512 1)
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MGE_CUDA_GENCODE}")
  if(MGE_WITH_TRT)
    include(cmake/tensorrt.cmake)
  endif()
  if(MGE_CUDA_USE_STATIC)
    if(MGE_WITH_TRT)
      if(MSVC OR WIN32)
        message(STATUS "windows TRT_LIBRARY: ${TRT_LIBRARY}")
        list(APPEND MGE_CUDA_LIBS ${TRT_LIBRARY} ${TRT_PLUGIN_LIBRARY})
      else()
        if(TensorRT_VERSION_MAJOR GREATER_EQUAL 8)
          list(APPEND MGE_CUDA_LIBS libnvinfer libnvinfer_plugin)
        else()
          list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libnvinfer libnvinfer_plugin
               -Wl,--no-whole-archive)
        endif()
      endif()
      if(TensorRT_VERSION_MAJOR STREQUAL 7)
        message(STATUS "handle trt myelin lib after trt7")
        list(APPEND MGE_CUDA_LIBS libmyelin_compiler libmyelin_executor
             libmyelin_pattern_runtime libmyelin_pattern_library)
      endif()
    endif()

    if("${CUDNN_VERSION}" STREQUAL "7.5.0")
      if(MSVC OR WIN32)
        message(STATUS "windows CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
        list(APPEND MGE_CUDA_LIBS ${CUDNN_LIBRARY})
      else()
        message(
          STATUS
            "cudnn 7.5.0 has bug in cudnnConvolutionBiasActivationForward, need --whole-archive to workaround, ref https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_7xx.html"
        )
        list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libcudnn -Wl,--no-whole-archive)
      endif()
    else()
      if(MSVC OR WIN32)
        message(STATUS "windows CUDNN_LIBRARY: ${CUDNN_LIBRARY}")
        list(APPEND MGE_CUDA_LIBS ${CUDNN_LIBRARY})
      else()
        list(APPEND MGE_CUDA_LIBS libcudnn)
      endif()
    endif()
    if(MSVC OR WIN32)
      list(APPEND MGE_CUDA_LIBS cusolver.lib curand.lib cudart_static.lib cusparse.lib)
    else()
      list(
        APPEND
        MGE_CUDA_LIBS
        cusolver_static
        curand_static
        culibos
        cudart_static
        cusparse_static)
    endif()
    if(MSVC OR WIN32)
      list(APPEND MGE_CUDA_LIBS cublas.lib)
    else()
      if(MGE_WITH_CUBLAS_SHARED)
        list(APPEND MGE_CUDA_LIBS cublas)
      else()
        list(APPEND MGE_CUDA_LIBS cublas_static)
      endif()
    endif()
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0"
       OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
      if(MSVC OR WIN32)
        list(APPEND MGE_CUDA_LIBS cublasLt.lib)
      else()
        if(MGE_WITH_CUBLAS_SHARED)
          list(APPEND MGE_CUDA_LIBS cublasLt)
        else()
          list(APPEND MGE_CUDA_LIBS cublasLt_static culibos)
        endif()
      endif()
    endif()
    if((${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0"
        OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
       AND NOT MSVC
       AND NOT WIN32)
      # mark all symbols from liblapack_static.a as weak to avoid duplicated definition
      # with mkl
      find_library(LAPACK_STATIC_PATH lapack_static
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
      set_target_properties(lapack_static_weak PROPERTIES IMPORTED_LOCATION
                                                          ${LAPACK_STATIC_COPY_PATH})
      list(APPEND MGE_CUDA_LIBS lapack_static_weak ${LAPACK_STATIC_COPY_PATH})
    endif()
  else()
    if(MGE_WITH_TRT)
      list(APPEND MGE_CUDA_LIBS libnvinfer libnvinfer_plugin)
      if(TensorRT_VERSION_MAJOR STREQUAL 7)
        message(STATUS "handle trt myelin lib after trt7")
        list(APPEND MGE_CUDA_LIBS libmyelin)
      endif()
    endif()
    list(APPEND MGE_CUDA_LIBS libcudnn)
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0"
       OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
      list(APPEND MGE_CUDA_LIBS cublasLt cusolver cublas curand)
    endif()
    list(APPEND MGE_CUDA_LIBS cudart)
  endif()

  if(NOT MGE_WITH_CUDA_STUB)
    if(MSVC OR WIN32)
      list(APPEND MGE_CUDA_LIBS cuda.lib)
    else()
      list(APPEND MGE_CUDA_LIBS cuda)
    endif()
  endif()

  if(NOT MGE_WITH_NVRTC_STUB)
    if(MSVC OR WIN32)
      list(APPEND MGE_CUDA_LIBS nvrtc.lib)
    else()
      list(APPEND MGE_CUDA_LIBS nvrtc)
    endif()
  endif()

  if(MGE_WITH_ANY_CUDA_STUB)
    add_subdirectory(dnn/cuda-stub)
    list(APPEND MGE_CUDA_LIBS cuda-stub)
  endif()

  if(MSVC OR WIN32)
    list(APPEND MGE_CUDA_LIBS nvrtc.lib)
  else()
    list(APPEND MGE_CUDA_LIBS nvToolsExt)
  endif()

  set(MGE_CUDA_LIBS "${MGE_CUDA_LIBS} -lrt")
  if(UNIX)
    set(MGE_CUDA_LIBS "${MGE_CUDA_LIBS} -ldl")
  endif()

endif()

# ##########please add_subdirectory from here###############
if((${MGE_ARCH} STREQUAL "x86_64"
    OR ${MGE_ARCH} STREQUAL "i386"
    OR ${MGE_ARCH} STREQUAL "armv7"
    OR ${MGE_ARCH} STREQUAL "aarch64"
   )
   AND NOT APPLE
   AND NOT MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
  option(MGE_ENABLE_CPUINFO "Build cpuinfo library for check runtime." ON)
  if(MGE_ENABLE_CPUINFO)
    message(STATUS "Enable cpuinfo runtime check and little kernel optimize.")
    add_definitions(-DMGB_ENABLE_CPUINFO_CHECK)
    include(cmake/cpuinfo.cmake)
  endif()
endif()

if(MGE_WITH_CAMBRICON)
  include_directories("$ENV{NEUWARE_HOME}/include")
  link_directories("$ENV{NEUWARE_HOME}/lib64")
  include(cmake/Cambricon/bang.cmake)
  include(cmake/Cambricon/cnrt.cmake)
  include(cmake/Cambricon/cndev.cmake)
  include(cmake/Cambricon/cndrv.cmake)
  list(APPEND MGE_CAMBRICON_LIBS libcnrt libcndev cnmlrt libcndrv)
  if(CNRT_VERSION_STRING VERSION_GREATER "5.0.0")
    include(cmake/Cambricon/cnnl.cmake)
    include(cmake/Cambricon/cnlight.cmake)
    include(cmake/Cambricon/magicmind.cmake)
    list(
      APPEND
      MGE_CAMBRICON_LIBS
      libcnnl
      libcnnl_extra
      libcnlight
      libmagicmind
      libmagicmind_runtime)
  else()
    include(cmake/cnml.cmake)
    list(APPEND MGE_CAMBRICON_LIBS libcnml)
  endif()
  set(MGE_CAMBRICON_LIBS "${MGE_CAMBRICON_LIBS}")
endif()

if(MGE_WITH_ROCM)
  include(cmake/rocm.cmake)
endif()

# TODO: add ascend.cmake
if(MGE_WITH_ATLAS)
  include_directories("$ENV{ASCEND_HOME_PATH}/include")
  link_directories("$ENV{ASCEND_HOME_PATH}/lib64")
  add_subdirectory(dnn/atlas-stub)
  list(APPEND MGE_ATLAS_LIBS atlas-stub)
  list(APPEND MGE_ATLAS_LIBS ascendcl nnopbase opapi acl_op_compiler)
  set(MGE_ATLAS_LIBS "${MGE_ATLAS_LIBS}")
  set(MGB_ATLAS ${MGE_WITH_ATLAS})
  set(MGB_USE_ATLAS_ASYNC_API ${MGE_WITH_ATLAS})
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
  message(
    STATUS
      "Exceptions disabled; MegEngine would kill itself when it is supposed to throw an exception."
  )
endif()
set(MGB_ENABLE_EXCEPTION ${MGE_ENABLE_EXCEPTIONS})
set(MEGDNN_ENABLE_EXCEPTIONS ${MGE_ENABLE_EXCEPTIONS})

# JIT
if(MGE_WITH_JIT AND MGE_WITH_HALIDE)
  set(HALIDE_SHARED_LIBRARY
      OFF
      CACHE BOOL "Build as a shared library")
  include(cmake/Halide.cmake)
endif()

if(MGE_ENABLE_EXCEPTIONS)
  add_compile_definitions(CPP_REDIS_ENABLE_EXCEPTION=1)
else()
  add_compile_definitions(CPP_REDIS_ENABLE_EXCEPTION=0)
endif()
include(cmake/cpp_redis.cmake)

# Thread
if(APPLE)
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  message(STATUS "disable jit, halide and mlir on macos host build...")
  set(MGE_WITH_HALIDE OFF)
  set(MGE_WITH_JIT OFF)
  set(MGE_WITH_JIT_MLIR OFF)
endif()

# riscv64
if(${MGE_ARCH} STREQUAL "riscv64")
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  message(STATUS "force config thread when build riscv64, as CMAKE detect failed")
endif()

set(MGB_JIT ${MGE_WITH_JIT})
set(MGB_JIT_MLIR ${MGE_WITH_JIT_MLIR})
set(MGB_JIT_HALIDE ${MGE_WITH_HALIDE})
# for consumer override MGB_C_OPR_INIT_FUNC symbol interface
if(NOT "${CUSTOM_C_OPR_INIT_FUNC}" STREQUAL "")
  add_compile_definitions(MGB_C_OPR_INIT_FUNC=${CUSTOM_C_OPR_INIT_FUNC})
  message(STATUS "override MGB_C_OPR_INIT_FUNC to ${CUSTOM_C_OPR_INIT_FUNC}")
endif()

set(MGB_CUSTOM_OP ${MGE_WITH_CUSTOM_OP})

if(MSVC OR WIN32)
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 1)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

if(CMAKE_THREAD_LIBS_INIT
   OR CMAKE_HAVE_LIBC_PTHREAD
   OR CMAKE_USE_WIN32_THREADS_INIT
   OR ANDROID
   OR OHOS)
  set(MGB_HAVE_THREAD 1)
endif()

if(MSVC OR WIN32)
  if(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
    message(STATUS "disable MGB_HAVE_THREAD/MGB_ENABLE_JSON when DEPLOY ON XP SP2")
    set(MGB_HAVE_THREAD 0)
    set(MGB_ENABLE_JSON 0)
  endif()
endif()

if(MGE_WITH_TEST)
  # use intra-op multi threads
  set(MEGDNN_ENABLE_MULTI_THREADS 1)
endif()

# benchmark
if(MGE_WITH_BENCHMARK)
  set(MEGDNN_WITH_BENCHMARK ${MGE_WITH_BENCHMARK})
endif()

# CUDA
set(MGB_CUDA ${MGE_WITH_CUDA})
set(MEGDNN_WITH_CUDA ${MGE_WITH_CUDA})

# ROCM
set(MGB_ROCM ${MGE_WITH_ROCM})
set(MEGDNN_WITH_ROCM ${MGE_WITH_ROCM})

# CAMBRICON
set(MGB_CAMBRICON ${MGE_WITH_CAMBRICON})
set(MEGDNN_WITH_CAMBRICON ${MGE_WITH_CAMBRICON})
# Debug info
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL
                                           "RelWithDebInfo")
  set(MGB_ASSERT_LOC 1)
  set(MGB_ENABLE_DEBUG_UTIL 1)
else()
  set(MGB_ASSERT_LOC 0)
  set(MGB_ENABLE_DEBUG_UTIL 0)
endif()

if(MSVC OR WIN32)
  if(${MGE_ARCH} STREQUAL "i386")
    set(MGB_ENABLE_DEBUG_UTIL 0)
    message(STATUS "disable MGB_ENABLE_DEBUG_UTIL at Windows i386 build")
  endif()
endif()

# TensorRT
set(MGB_ENABLE_TENSOR_RT ${MGE_WITH_TRT})

# inference need jit now, also keep same build logic with bazel
if(MGE_DEPLOY_INFERENCE_ON_WINDOWS_XP_SP2)
  set(MGB_BUILD_SLIM_SERVING 1)
else()
  set(MGB_BUILD_SLIM_SERVING 0)
endif()
# Inference only
if(MGE_INFERENCE_ONLY AND NOT MGE_WITH_TEST)
  set(MGB_ENABLE_GRAD 0)
else()
  set(MGB_ENABLE_GRAD 1)
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
# dotprod is not enable by default on APPLE, cpuinfo has some problem on APPLE
if(NOT APPLE AND ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
  check_cxx_compiler_flag("-march=armv8.2-a+dotprod" CXX_COMPILER_SUPPORT_DOT)
  if(CXX_COMPILER_SUPPORT_DOT)
    message(STATUS "Enable dotprod feature in armv8.2-a using MGB_ENABLE_DOT")
    set(MGB_ENABLE_DOT 1)
  endif()
endif()

if(MGE_ARCH STREQUAL "armv7")
  # -funsafe-math-optimizations to enable neon auto-vectorization (since neon is not
  # fully IEEE 754 compatible, GCC does not turn on neon auto-vectorization by default.
  if(ANDROID OR OHOS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=softfp -mfpu=neon")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funsafe-math-optimizations")
  set(MARCH "-march=armv7-a")
  set(MEGDNN_ARMV7 1)
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

  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(
      WARNING
        "aarch64 ld will add -mfix-cortex-a53-843419 and -mfix-cortex-a53-835769 by default.\
    when build with DEBUG build type,ld will take about 14min+, for save link time(14min->1min), \
    you may open below flags if not deploy on arm a53 platform, or just build release type!"
    )
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mno-fix-cortex-a53-843419
    # -mno-fix-cortex-a53-835769")
  endif()

  if(MGE_WITH_CUDA)
    message(STATUS "check compiler version...")
    if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.5)
      message(FATAL_ERROR "gcc version must >= 7.5 when build with cuda")
    endif()
    # cuda libs build with -mno-fix-cortex-a53-843419 -mno-fix-cortex-a53-835769, if you are not deploy on arm a53 platform and want to save link time, you may open below flags even
    message(
      WARNING
        "cuda libs build with -mno-fix-cortex-a53-843419 -mno-fix-cortex-a53-835769, so force disable it to avoid link error"
    )
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -mno-fix-cortex-a53-843419 -mno-fix-cortex-a53-835769")
  endif()
endif()

if(MGE_ARCH STREQUAL "riscv64")
  set(MEGDNN_RISCV64 1)
  set(MEGDNN_64_BIT 1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH}")

set(MGE_VERSION_SCRIPT
    ${PROJECT_SOURCE_DIR}/src/version.ld
    CACHE INTERNAL "Path to linker version script")

execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_FULL_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Write out megbrain_build_config.h It defines macros needed by both megbrain and dnn
# please don't put the configuration that is easy to change at
# megbrain_build_config.h.in for example cuda_sm_gen.h.in and git_full_hash_header.h.in,
# which will lead to CMake build dirty file issue
configure_file(src/megbrain_build_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(MGE_WITH_CUDA)
  configure_file(src/cuda_sm_gen.h.in
                 ${CMAKE_CURRENT_BINARY_DIR}/genfiles/cuda_sm_gen.h)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/cuda_sm_gen.h
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

configure_file(src/git_full_hash_header.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/genfiles/git_full_hash_header.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/git_full_hash_header.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(dnn)

list(APPEND MGB_OPR_PARAM_DEFS_SRCS
     ${CMAKE_CURRENT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py)
set(MGB_OPR_PARAM_DEFS_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/dnn/scripts/gen_param_defs.py)

set(MGB_OPR_PARAM_DEFS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/opr/include/)
file(MAKE_DIRECTORY ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr)
add_custom_command(
  OUTPUT ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
  COMMAND
    ${PYTHON3_EXECUTABLE_WITHOUT_VERSION} ${MGB_OPR_PARAM_DEFS_SCRIPT}
    ${MGB_OPR_PARAM_DEFS_SRCS} ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
  DEPENDS ${MGB_OPR_PARAM_DEFS_SRCS} ${MGB_OPR_PARAM_DEFS_SCRIPT}
  VERBATIM)

list(APPEND MGB_OPR_PARAM_DEFS_OUTS
     ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h)

install(FILES ${MGB_OPR_PARAM_DEFS_OUTS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/megbrain/opr/)

list(APPEND MGB_OPR_PARAM_DEFS_INC ${MGB_OPR_PARAM_DEFS_OUT_DIR})
add_custom_target(_mgb_opr_param_defs DEPENDS ${MGB_OPR_PARAM_DEFS_OUTS})
add_library(mgb_opr_param_defs INTERFACE)
target_include_directories(
  mgb_opr_param_defs INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                               $<BUILD_INTERFACE:${MGB_OPR_PARAM_DEFS_INC}>)
add_dependencies(mgb_opr_param_defs _mgb_opr_param_defs)
install(TARGETS mgb_opr_param_defs EXPORT ${MGE_EXPORT_TARGETS})

if(MGE_WITH_JIT_MLIR OR MGE_BUILD_IMPERATIVE_RT)
  # generate param_defs.td
  set(MGE_GENFILE_DIR ${PROJECT_BINARY_DIR}/src/genfiles)
  set(MGE_GEN_IR_DIR ${PROJECT_BINARY_DIR}/src/core/include/megbrain/ir)
  set(OPR_PARAM_DEFS_SRCS ${MGE_GENFILE_DIR}/opr_param_defs.py)
  set(OPR_PARAM_DEFS_SCRIPT ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_tablegen.py)
  set(OPR_PARAM_DEFS_OUT ${MGE_GEN_IR_DIR}/param_defs.td)
  file(COPY ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py
       DESTINATION ${MGE_GENFILE_DIR})
  file(READ ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py CONTENTS)
  file(APPEND ${OPR_PARAM_DEFS_SRCS} ${CONTENTS})
  file(MAKE_DIRECTORY ${MGE_GEN_IR_DIR})
  add_custom_command(
    OUTPUT ${OPR_PARAM_DEFS_OUT}
    COMMAND ${PYTHON3_EXECUTABLE_WITHOUT_VERSION} ${OPR_PARAM_DEFS_SCRIPT}
            ${OPR_PARAM_DEFS_SRCS} ${OPR_PARAM_DEFS_OUT}
    DEPENDS ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py
            ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py
            ${OPR_PARAM_DEFS_SCRIPT}
    VERBATIM)
  # mlir tblgen sources
  set(MGE_IR_DIR ${PROJECT_SOURCE_DIR}/src/core/include/megbrain/ir)
  set(MGE_IR_INCLUDE_DIRS ${MLIR_LLVM_INCLUDE_DIR} ${MGE_IR_DIR} ${MGE_GEN_IR_DIR})
  list(TRANSFORM MGE_IR_INCLUDE_DIRS PREPEND "-I")
  file(GLOB_RECURSE MGE_IR_TDS ${MGE_IR_DIR}/*.td)
  add_custom_target(param_defs_tblgen DEPENDS ${OPR_PARAM_DEFS_OUT})
endif()

if(MGE_WITH_DISTRIBUTED)
  set(MEGRAY_WITH_NCCL
      ${MGE_WITH_CUDA}
      CACHE BOOL "Override MegRay option" FORCE)
  set(MEGRAY_WITH_SHM
      ${MGE_WITH_CUDA}
      CACHE BOOL "Override MegRay option" FORCE)
  set(MEGRAY_WITH_RCCL
      ${MGE_WITH_ROCM}
      CACHE BOOL "Override MegRay option" FORCE)
  set(MEGRAY_WITH_CNCL
      ${MGE_WITH_CAMBRICON}
      CACHE BOOL "Override MegRay option" FORCE)
  set(MEGRAY_WITH_HCCL
      ${MGE_WITH_ATLAS}
      CACHE BOOL "Override MegRay option" FORCE)
  set(MEGRAY_CUDA_GENCODE
      ${MGE_CUDA_GENCODE}
      CACHE STRING "Overwrite MegRay CUDA -gencode specifications" FORCE)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/MegRay)
endif()

add_subdirectory(src)

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
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/imperative/python/${PACKAGE_NAME}/core/$<TARGET_FILE_NAME:${MODULE_NAME}>
      ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/core/$<TARGET_FILE_NAME:${MODULE_NAME}>
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/imperative/python/${PACKAGE_NAME}/version.py
      ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/version.py
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/src/custom/include
      ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/core/include
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/core/lib
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_BINARY_DIR}/src/$<TARGET_FILE_NAME:${MGE_SHARED_LIB}>
      ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python/${PACKAGE_NAME}/core/lib/$<TARGET_FILE_NAME:${MGE_SHARED_LIB}>
    DEPENDS ${develop_depends}
    VERBATIM)
  add_dependencies(develop _imperative_rt)

  # generate stub file for _imperative_rt
  execute_process(
    COMMAND ${PYTHON3_EXECUTABLE_WITHOUT_VERSION} -c
            "import mypy.version; assert mypy.version.__version__ >= '0.982'"
    RESULT_VARIABLE NOT_HAVING_MYPY_STUBGEN)
  if(NOT ${NOT_HAVING_MYPY_STUBGEN})
    add_custom_command(
      TARGET develop
      POST_BUILD
      COMMAND
        ${PYTHON3_EXECUTABLE_WITHOUT_VERSION} -c "from mypy.stubgen import main; main()"
        -p ${PACKAGE_NAME}.core.${MODULE_NAME} -o
        ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/imperative/python
      VERBATIM)
  endif()
endif()

# Configure and install pkg-config. Note that unlike the Config.cmake modules, this is
# not relocatable (and not really portable) because we have two dependencies without
# pkg-config descriptions: FlatBuffers and MKL-DNN
if(MGE_USE_SYSTEM_MKLDNN)
  set(MGE_PKGCONFIG_LIBS_PRIVATE "-ldnnl")
endif()
if(MGE_USE_SYSTEM_OPENBLAS)
  set(MGE_PKGCONFIG_LIBS_PRIVATE "${MGE_PKGCONFIG_LIBS_PRIVATE} -lopenblas")
endif()
configure_file(cmake/megengine.pc.in ${CMAKE_CURRENT_BINARY_DIR}/megengine.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/megengine.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# Do not export targets if MGE_WITH_DISTRIBUTED is on. MegRay is not ready.
if(NOT MGE_WITH_DISTRIBUTED)
  include(CMakePackageConfigHelpers)
  set(MGE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/MegEngine)
  configure_package_config_file(
    cmake/MegEngineConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfig.cmake
    INSTALL_DESTINATION ${MGE_INSTALL_CMAKEDIR})
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfigVersion.cmake
    VERSION ${MGB_VER_STRING}
    COMPATIBILITY SameMajorVersion)

  install(EXPORT ${MGE_EXPORT_TARGETS} DESTINATION ${MGE_INSTALL_CMAKEDIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfig.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/MegEngineConfigVersion.cmake
          DESTINATION ${MGE_INSTALL_CMAKEDIR})
endif()

if(MGE_WITH_JIT_MLIR)
  add_subdirectory(tools/mlir/mgb-opt)
  add_subdirectory(tools/mlir/mgb-file-check)
endif()

if(MGE_WITH_CUDA
   AND MGE_CUDA_USE_STATIC
   AND ("${CUDNN_VERSION}" VERSION_GREATER "8.0.0" OR "${CUDNN_VERSION}" VERSION_EQUAL
                                                      "8.0.0")
   AND (NOT MGE_WITH_CUDNN_SHARED))
  message(
    WARNING
      "Static link CUDNN8 with many sm is unworkable, please use -DMGE_WITH_CUDNN_SHARED=ON or -DMGE_WITH_LARGE_ARCHIVE=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_70,code=sm_70 arch=compute_75,code=sm_75\" "
  )
  message(
    WARNING
      "Static link CUDNN8 with many sm is unworkable, please use -DMGE_WITH_CUDNN_SHARED=ON or -DMGE_WITH_LARGE_ARCHIVE=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_70,code=sm_70 arch=compute_75,code=sm_75\" "
  )
  message(
    WARNING
      "Static link CUDNN8 with many sm is unworkable, please use -DMGE_WITH_CUDNN_SHARED=ON or -DMGE_WITH_LARGE_ARCHIVE=ON -DMGE_CUDA_GENCODE=\"-gencode arch=compute_70,code=sm_70 arch=compute_75,code=sm_75\" "
  )
endif()

if(MGE_WITH_LITE)
  add_subdirectory(lite)
endif()

if(ANDROID)
  message(
    WARNING
      "MegEngine project use thread_local, if you want to deploy MegEngine at dlopen/dlclose scene, please build with c++_shared by -DANDROID_STL=c++_shared, detail at https://github.com/android-ndk/ndk/issues/789 for example: EXTRA_CMAKE_ARGS=\" -DANDROID_STL=c++_shared\" ./scripts/cmake-build/cross_build_android_arm_inference.sh "
  )
endif()
