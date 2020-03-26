# MegEngine Tests

* unit: This directory has the same layout as megengine directory.
* regression: Small tests to check whether the old issue is fixed.
* integration: Tests involve multiple parts of megengine, tests that longer than 1min should be a manual test.
* pytorch_comparison: Special directory for torch-related test
* helpers
    - Test utilities should placed in this directory
    - `from helpers import ...` in your test code


## Default running setup

Execute `run.sh` to test default set of tests.

- No torch related test
- No internet related test
- No doc related test
