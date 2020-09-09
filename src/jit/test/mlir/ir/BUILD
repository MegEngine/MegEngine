load("//brain/megbrain/src/jit/test/mlir/utils:lit.bzl", "mlir_lit_test_suite")

filegroup(
    name = "mlir_test_tools",
    testonly = True,
    data = [
        "//brain/megbrain/tools/mlir:mgb-opt",
        "//brain/megbrain/tools/mlir:mgb-file-check"
    ],
)

mlir_lit_test_suite(
    name = "mlir_pass_check",
    data = [":mlir_test_tools"],
    test_file_exts = ["mlir",]
)
