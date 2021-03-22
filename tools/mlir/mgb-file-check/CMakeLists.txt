add_custom_command(
    OUTPUT link_sh
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/tools/mlir/mgb-file-check/mgb-file-check.sh
        ${PROJECT_BINARY_DIR}/tools/mlir/mgb-file-check/mgb-file-check
)

add_custom_target(mgb-file-check DEPENDS link_sh)