configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/utils/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/lit.cfg.py
)

set(LLVM_EXTERNAL_LIT "${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/utils/lit/lit.py" CACHE STRING "External lit")

set(MLIR_MGB_TEST_DEPENDS
        mgb-file-check
        count not
        mgb-opt
)

add_lit_testsuite(mgb-mlir-test-lit "Running the mgb regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}/utils
        DEPENDS ${MLIR_MGB_TEST_DEPENDS}
        )
set_target_properties(mgb-mlir-test-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(MLIR_TEST ${CMAKE_CURRENT_SOURCE_DIR} 
        DEPENDS ${MLIR_MGB_TEST_DEPENDS}
)

add_custom_target(mlir_pass_check)
add_dependencies(mlir_pass_check mgb-mlir-test-lit)
