set(SOURCES
    ../../dnn/scripts/opr_param_defs.py
    ../../src/core/include/megbrain/ir/ops.td
    generated/opdef.h.inl
    generated/opdef.cpp.inl
    generated/opdef.py.inl
    generated/opdef.cpy.inl
    generated/enum_macro.h)
execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum ${SOURCES}
                OUTPUT_VARIABLE GENERATED_HASH_CONTENT)

file(READ generated/hash.txt HASH_CONTENT)

if(NOT "${GENERATED_HASH_CONTENT}" STREQUAL "${HASH_CONTENT}")
  message(FATAL_ERROR "File ops.td was changed, please rerun cmake configure")
endif()
