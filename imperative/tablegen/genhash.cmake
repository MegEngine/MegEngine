set(SOURCES
    ../../dnn/scripts/opr_param_defs.py
    ../../src/core/include/megbrain/ir/ops.td
    generated/opdef.h.inl
    generated/opdef.cpp.inl
    generated/opdef.py.inl
    generated/opdef.cpy.inl
    generated/enum_macro.h)
execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum ${SOURCES}
                OUTPUT_VARIABLE HASH_CONTENT)
message(STATUS "Generating hash.txt for opdefs")
file(WRITE generated/hash.txt "${HASH_CONTENT}")
