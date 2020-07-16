cc_library(
    name = "mgblar",
    copts = ["-std=c++14"],
    srcs = [
        "src/infile_persistent_cache.cpp",
        "src/mgblar.cpp",
        "src/json_loader.cpp",
    ],
    hdrs = [
        "src/infile_persistent_cache.h",
        "src/mgblar.h",
        "src/json_loader.h",
        "src/npy.h",
    ],
    features = if_opt([
        "no_exceptions",
        "no_rtti",
    ]),
    includes = ["src"],
    defines = ["MGB_ENABLE_FASTRUN=1"],
    deps = ["//brain/megbrain:sdk-test"],
)

cc_megvii_binary(
    name = "load_and_run",
    copts = ["-std=c++14"],
    srcs = ["main.cpp"],
    features = if_opt([
        "no_exceptions",
        "no_rtti",
    ]),
    internal_deps = [":mgblar"],
    visibility = ["//visibility:public"],
)

cc_megvii_shared_object(
    name = "load_and_run_shared",
    copts = ["-std=c++14"],
    srcs = ["main.cpp"],
    features = if_opt([
        "no_exceptions",
        "no_rtti",
    ]),
    internal_deps = [":mgblar"],
    syms = ["main"],
)

cc_megvii_binary(
    name = "json_loader_test",
    copts = ["-std=c++14"],
    srcs = ["test/json_loader_test.cpp"],
    internal_deps = [":mgblar"],
)

cc_library(
    name = "megbrain_ios_lar_lib",
    srcs = [
        "src/infile_persistent_cache.cpp",
        "src/mgblar.cpp",
    ],
    hdrs = [
        "src/infile_persistent_cache.h",
        "src/mgblar.h",
    ],
    copts = ["-DMGB_NO_MAIN=1"],
    features = if_opt([
        "no_exceptions",
        "no_rtti",
    ]),
    deps = ["//brain/megbrain:sdk-test"],
)

cc_megvii_static_library(
    name = "megbrain_ios_lar",
    deps = [":megbrain_ios_lar_lib"],
)
