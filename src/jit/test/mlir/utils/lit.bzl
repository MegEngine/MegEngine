# Test definitions for Lit, the LLVM test runner.
#
"""Lit runner globbing test
"""

# Default values used by the test runner.
_default_test_file_exts = ["mlir", "pbtxt", "td"]
_default_size = "small"
_default_tags = []

# These are patterns which we should never match, for tests, subdirectories, or
# test input data files.
_ALWAYS_EXCLUDE = [
    "**/LICENSE.txt",
    "**/README.txt",
    "**/lit.local.cfg",
    # Exclude input files that have spaces in their names, since bazel
    # cannot cope with such "targets" in the srcs list.
    "**/* *",
    "**/* */**",
]

def _run_lit_test(name, data, size, tags, features):
    """Runs lit on all tests it can find in `data` under megbrain/src/jit/test/mlir/ir.

    Note that, due to Bazel's hermetic builds, lit only sees the tests that
    are included in the `data` parameter, regardless of what other tests might
    exist in the directory searched.

    Args:
      name: str, the name of the test, including extension.
      data: [str], the data input to the test.
      size: str, the size of the test.
      tags: [str], tags to attach to the test.
      features: [str], list of extra features to enable.
    """

    native.py_test(
        name = name,
        srcs = ["@llvm-project//llvm:lit"],
        tags = tags,
        args = [
            "brain/megbrain/src/jit/test/mlir/utils --config-prefix=lit.bzl -v",
        ] + features,
        data = data + [
            "//brain/megbrain/src/jit/test/mlir/utils:litfiles",
            "//brain/megbrain/tools/mlir:mgb-file-check",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:not",
        ],
        size = size,
        main = "lit.py",
    )

def mlir_lit_test_suite(
        name,
        exclude = [],
        test_file_exts = _default_test_file_exts,
        default_size = _default_size,
        size_override = {},
        data = [],
        per_test_extra_data = {},
        default_tags = _default_tags,
        tags_override = {},
        features = []):
    """Creates all plausible Lit tests (and their inputs) under this directory.

    Args:
      name: str, name of the generated test suite.
      exclude: [str], paths to exclude (for tests and inputs).
      test_file_exts: [str], extensions for files that are tests.
      default_size: str, the test size for targets not in "size_override".
      size_override: {str: str}, sizes to use for specific tests.
      data: [str], additional input data to the test.
      per_test_extra_data: {str: [str]}, extra data to attach to a given file.
      default_tags: [str], additional tags to attach to the test.
      tags_override: {str: str}, tags to add to specific tests.
      features: [str], list of extra features to enable.
    """

    # Ignore some patterns by default for tests and input data.
    exclude = _ALWAYS_EXCLUDE + exclude

    test_names = []
    tests = native.glob(
        ["*." + ext for ext in test_file_exts],
        exclude = exclude,
    )

    # Run tests individually such that errors can be attributed to a specific
    # failure.
    for i in range(len(tests)):
        cur_test = tests[i]

        # Instantiate this test with updated parameters.
        internal_name = cur_test
        lit_test(
            name = internal_name,
            data = data + per_test_extra_data.pop(cur_test, []),
            size = size_override.pop(cur_test, default_size),
            tags = ["windows_fail"] + default_tags + tags_override.pop(cur_test, []),
            features = features,
        )
        test_names.append(internal_name + ".test")

    native.test_suite(
        name = name,
        tests = test_names,
        tags = default_tags,
    )

def lit_test(
        name,
        data = [],
        size = _default_size,
        tags = _default_tags,
        features = []):
    """Runs test files under lit.

    Args:
      name: str, the name of the test.
      data: [str], labels that should be provided as data inputs.
      size: str, the size of the test.
      tags: [str], tags to attach to the test.
      features: [str], list of extra features to enable.
    """
    _run_lit_test(name + ".test", data + [name], size, tags, features)
