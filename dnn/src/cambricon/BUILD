load("//brain/megbrain/dnn:flags.bzl", "megdnn_opts")
load("@megvii3//tools/build_rules:bangc.bzl", "bangc_library")

package(default_visibility = ["//brain/megbrain/dnn:__subpackages__"])

bangc_library(
    name = "bangc_kernels",
    srcs = glob([
        "**/*.mlu",
    ]) + [
        "//brain/megbrain/dnn:src/common/utils.cuh",
    ],
    hdrs = glob([
        "**/*.mlu.h",
    ]),
    deps = [
        "//brain/megbrain/dnn:public_headers",
    ],
    copts = megdnn_opts + [
        "-Ibrain/megbrain/dnn",
    ],
)

filegroup(
    name = "cambricon_backend_files",
    srcs = glob([
        "**/*.cpp",
        "**/*.h",
        "**/*.hpp",
    ]),
)
