set(SYNC_THIRD_PARTY_CMD "bash")
list(APPEND SYNC_THIRD_PARTY_CMD "${CMAKE_CURRENT_SOURCE_DIR}/third_party/prepare.sh")

if(NOT MGE_WITH_JIT_MLIR AND NOT MGE_BUILD_IMPERATIVE_RT)
  list(APPEND SYNC_THIRD_PARTY_CMD "-a")
endif()

if(NOT MGE_WITH_TEST)
  list(APPEND SYNC_THIRD_PARTY_CMD "-b")
endif()

if((NOT MGE_WITH_MKLDNN) OR (NOT ${MGE_ARCH} STREQUAL "x86_64"))
  list(APPEND SYNC_THIRD_PARTY_CMD "-c")
endif()

if(NOT MGE_WITH_HALIDE)
  list(APPEND SYNC_THIRD_PARTY_CMD "-d")
endif()

if(NOT MGE_WITH_DISTRIBUTED)
  list(APPEND SYNC_THIRD_PARTY_CMD "-e")
  list(APPEND SYNC_THIRD_PARTY_CMD "-i")
endif()

if(NOT MGE_WITH_CUDA)
  list(APPEND SYNC_THIRD_PARTY_CMD "-f")
endif()

if(NOT MGE_BUILD_IMPERATIVE_RT)
  list(APPEND SYNC_THIRD_PARTY_CMD "-g")
endif()

if(NOT ${MGE_BLAS} STREQUAL "OpenBLAS")
  list(APPEND SYNC_THIRD_PARTY_CMD "-j")
endif()

message("sync third_party with command: ${SYNC_THIRD_PARTY_CMD}")
execute_process(
  COMMAND ${SYNC_THIRD_PARTY_CMD}
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE sync_third_party_cmd_ret)
if(NOT sync_third_party_cmd_ret EQUAL 0)
  message(
    FATAL_ERROR "Error run sync third_party please run ${SYNC_THIRD_PARTY_CMD} manually"
  )
endif()
