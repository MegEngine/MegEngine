load("//brain/megbrain/lite:flags.bzl","pthread_select", "lite_opts")
cc_library(
    name = "mgblar",
    srcs = glob(["src/**/*.cpp"], exclude = ["src/main.cpp"]),
    hdrs = glob(["src/**/*.h"]),
    includes = ["src"],
    features = if_opt([
        "no_rtti",
    ]),

    deps = ["//brain/megbrain/lite:lite_static_test"]+
        pthread_select(
            ["@com_github_gflags_gflags//:gflags_nothreads"],
            ["//external:gflags"]
        ),

    copts = ["-std=c++14"] + lite_opts,
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_megvii_binary(
    name = "load_and_run",
    copts = ["-std=c++14"],
    srcs = ["src/main.cpp"],
    features = if_opt([
        "no_exceptions",
        "no_rtti",
    ]),
    internal_deps = [":mgblar"],
    visibility = ["//visibility:public"],
)

