

file(GLOB_RECURSE SOURCES core/impl/*.cpp gopt/impl/*.cpp opr/impl/*.cpp plugin/impl/*.cpp serialization/impl/*.cpp core/impl/*.inl gopt/impl/*.inl opr/impl/*.inl plugin/impl/*.inl serialization/impl/*.inl)

set(MGB_DEF)
if(MGE_WITH_JIT)
    list(APPEND MGB_DEF -DMGB_JIT=1)
    file(GLOB_RECURSE SOURCES_ jit/impl/*.cpp jit/impl/*.inl)
    list(APPEND SOURCES ${SOURCES_})
else()
    list(APPEND MGB_DEF -DMGB_JIT=0)
endif()

if(MGE_INFERENCE_ONLY)
    list(APPEND MGB_DEF -DMGB_ENABLE_GRAD=0 -DMGB_BUILD_SLIM_SERVING=1)
else()
    list(APPEND MGB_DEF -DMGB_ENABLE_GRAD=1 -DMGB_BUILD_SLIM_SERVING=0)
endif()

if(MGE_WITH_DISTRIBUTED)
    file(GLOB_RECURSE SOURCES_ opr-mm/impl/*.cpp opr-mm/impl/*.inl)
    list(APPEND SOURCES ${SOURCES_})
endif()

set(MGB_INC core/include gopt/include opr/include plugin/include serialization/include)

if(MGE_WITH_JIT)
    list(APPEND MGB_INC jit/include)
    if(MGE_WITH_CUDA)
        list(APPEND MGB_INC jit/impl/cuda)
    endif()
endif()

if(MGE_WITH_DISTRIBUTED)
    list(APPEND MGB_DEF -DMGB_ENABLE_OPR_MM=1)
    list(APPEND MGB_INC opr-mm/include)
else()
    list(APPEND MGB_DEF -DMGB_ENABLE_OPR_MM=0)
endif()

if(MGE_WITH_CUDA)
list(APPEND MGB_DEF -DMGB_CUDA=1)
else()
list(APPEND MGB_DEF -DMGB_CUDA=0)
endif()

if(MGE_WITH_CUDA AND MGE_WITH_TRT)
    list(APPEND MGB_DEF -DMGB_ENABLE_TENSOR_RT=1)
    list(APPEND MGB_INC tensorrt/include)
    file(GLOB_RECURSE SOURCES_ tensorrt/impl/*.cpp tensorrt/impl/*.inl)
    list(APPEND SOURCES ${SOURCES_})
else()
    list(APPEND MGB_DEF -DMGB_ENABLE_TENSOR_RT=0)
endif()

set(MGB_DEF ${MGB_DEF} PARENT_SCOPE)
add_library(megbrain STATIC EXCLUDE_FROM_ALL ${SOURCES})
target_link_libraries(megbrain mgb_opr_param_defs)
target_compile_definitions(megbrain PUBLIC ${MGB_DEF})
target_include_directories(megbrain PUBLIC ${MGB_INC})

if(MGE_WITH_CUDA)
    target_compile_options(megbrain PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>"
        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unused-parameter>")
else()
    target_compile_options(megbrain PRIVATE "-Wno-unused-parameter")
endif()
if(CXX_SUPPORT_WCLASS_MEMACCESS)
    if(MGE_WITH_CUDA)
        target_compile_options(megbrain PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-class-memaccess>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-class-memaccess>")
    else()
        target_compile_options(megbrain PRIVATE "-Wno-class-memaccess")
    endif()
endif()
target_link_libraries(megbrain megdnn)
if(MGE_WITH_DISTRIBUTED)
    target_link_libraries (megbrain megray)
endif()
target_link_libraries(megbrain ${MGE_CUDA_LIBS})
if(MGE_WITH_JIT AND MGE_WITH_HALIDE)
    target_link_libraries(megbrain libhalide)
    target_link_libraries(megbrain ${HALIDE_LLVM_LIBS})
endif()
if (MGB_WITH_FLATBUFFERS)
    set (GEN_FLATBUFFERS_SCHEMA_PY ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_flatbuffers_schema.py)
    set (OPR_PARAM_DEFS_PY ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py)
    set (MGB_PARAM_DEFS_PY ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py)
    file (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl)
    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/opr_param_defs.fbs
        COMMAND
            ${PYTHON_EXECUTABLE} ${GEN_FLATBUFFERS_SCHEMA_PY} ${OPR_PARAM_DEFS_PY} ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/opr_param_defs.fbs
        DEPENDS ${GEN_FLATBUFFERS_SCHEMA_PY} ${OPR_PARAM_DEFS_PY} 
        VERBATIM
    )
    add_custom_command(
        OUTPUT
            ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/mgb_opr_param_defs.fbs
        COMMAND
            ${PYTHON_EXECUTABLE} ${GEN_FLATBUFFERS_SCHEMA_PY} ${MGB_PARAM_DEFS_PY} ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/mgb_opr_param_defs.fbs
        DEPENDS ${GEN_FLATBUFFERS_SCHEMA_PY} ${MGB_PARAM_DEFS_PY}
        VERBATIM
    )
    list(APPEND FLATBUFFERS_SCHEMA_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/impl/dtype.fbs
        ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/opr_param_defs.fbs
        ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl/mgb_opr_param_defs.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/opr/impl/mgb_cpp_opr.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/impl/schema.fbs
    )
    list(APPEND FLATBUFFERS_SCHEMA_INCLUDE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/impl
        ${CMAKE_CURRENT_BINARY_DIR}/serialization/impl
        ${CMAKE_CURRENT_SOURCE_DIR}/opr/impl
    )
    build_flatbuffers(
        "${FLATBUFFERS_SCHEMA_FILES}"
        "${FLATBUFFERS_SCHEMA_INCLUDE_DIR}"
        mgb_serialization_schema_fbs
        "${FLATBUFFERS_SCHEMA_FILES}"
        "${CMAKE_CURRENT_BINARY_DIR}/serialization/include/megbrain/serialization/internal"
        ""
        ""
    )
    add_dependencies(megbrain mgb_serialization_schema_fbs)
    target_include_directories(megbrain PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/serialization/include)
    target_compile_definitions(megbrain PUBLIC MGB_ENABLE_FBS_SERIALIZATION=1)
    target_link_libraries(megbrain flatbuffers)
    
    set (GENERATED_FLATBUFFERS_CONVERTER_PATH ${CMAKE_CURRENT_BINARY_DIR}/genfiles/this/should/be/added/to/sereg_caller/target/only/but/hey/yolo/there/are/no/fine/grained/targets/let/us/just/make/this/path/unreasonably/long/to/avoid/collision)
    set (GEN_FLATBUFFERS_CONVERTER_PY ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_flatbuffers_converter.py)
    file (MAKE_DIRECTORY ${GENERATED_FLATBUFFERS_CONVERTER_PATH})
    add_custom_command(
        OUTPUT
            ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/opr_param_defs_converter.inl
        COMMAND
            ${PYTHON_EXECUTABLE} ${GEN_FLATBUFFERS_CONVERTER_PY} ${OPR_PARAM_DEFS_PY} ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/opr_param_defs_converter.inl
        DEPENDS ${GEN_FLATBUFFERS_CONVERTER_PY} ${OPR_PARAM_DEFS_PY}
        VERBATIM
    )
    add_custom_command(
        OUTPUT
            ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/mgb_opr_param_defs_converter.inl
        COMMAND
            ${PYTHON_EXECUTABLE} ${GEN_FLATBUFFERS_CONVERTER_PY} ${MGB_PARAM_DEFS_PY} ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/mgb_opr_param_defs_converter.inl
        DEPENDS ${GEN_FLATBUFFERS_CONVERTER_PY} ${MGB_PARAM_DEFS_PY}
        VERBATIM
    )
    target_sources(megbrain PRIVATE ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/opr_param_defs_converter.inl)
    target_sources(megbrain PRIVATE ${GENERATED_FLATBUFFERS_CONVERTER_PATH}/mgb_opr_param_defs_converter.inl)
    target_include_directories(megbrain PRIVATE ${GENERATED_FLATBUFFERS_CONVERTER_PATH})
endif()
if(UNIX)
    target_link_libraries(megbrain dl rt)
    find_library(libatomic atomic)
    if(NOT libatomic STREQUAL "libatomic-NOTFOUND")
        target_link_libraries(megbrain libatomic)
    endif()
endif()

add_library(megengine SHARED $<TARGET_PROPERTY:megbrain,SOURCES>)
if(MGB_WITH_FLATBUFFERS)
    add_dependencies(megengine mgb_serialization_schema_fbs)
endif()
target_include_directories(megengine PRIVATE $<TARGET_PROPERTY:megbrain,INCLUDE_DIRECTORIES>)
target_compile_definitions(megengine PUBLIC $<TARGET_PROPERTY:megbrain,COMPILE_DEFINITIONS>)
target_compile_options(megengine PUBLIC $<TARGET_PROPERTY:megbrain,COMPILE_OPTIONS>)
target_link_libraries(megengine PRIVATE $<TARGET_PROPERTY:megbrain,LINK_LIBRARIES> -Wl,--no-undefined -Wl,--version-script=${PROJECT_SOURCE_DIR}/python_module/src/version.ld)
set_target_properties(megengine PROPERTIES CXX_VISIBILITY_PRESET default)
install(TARGETS megengine LIBRARY DESTINATION lib64)
foreach(_PATH ${MGB_INC})
    install(DIRECTORY ${_PATH}/megbrain DESTINATION include FILES_MATCHING PATTERN "*.h")
endforeach()
