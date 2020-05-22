# Load and Run

Load a model and run, for testing/debugging/profiling.

## Build

<!--
-->

### Build with cmake

Build MegEngine from source following [README.md](../../README.md). It will also produce the executable, `load_and_run`, which loads a model and runs the test cases attached to the model.


<!--
-->

## Dump Model with Test Cases Using [dump_with_testcase_mge.py](dump_with_testcase_mge.py)

### Step 1

Dump the model by calling the python API `megengine.jit.trace.dump()`.

### Step 2

Append the test cases to the dumped model using [dump_with_testcase_mge.py](dump_with_testcase_mge.py).

The basic usage of [dump_with_testcase_mge.py](dump_with_testcase_mge.py) is

```
python3 dump_with_testcase_mge.py model -d input_description -o model_with_testcases

```

where `model` is the file dumped at step 1, `input_description` describes the input data of the test cases, and `model_with_testcases` is the saved model with test cases.

`input_description` can be provided in the following ways:

1. In the format `var0:file0;var1:file1...` meaning that `var0` should use
   image file `file0`, `var1` should use image `file1` and so on. If there
   is only one input var, the var name can be omitted. This can be combined
   with `--resize-input` option.
2. In the format `var0:#rand(min, max, shape...);var1:#rand(min, max)...`
   meaning to fill the corresponding input vars with uniform random numbers
   in the range `[min, max)`, optionally overriding its shape.

For more usages, run

```
python3 dump_with_testcase_mge.py --help
```

### Example

1. Obtain the model file by running [xornet.py](../xor-deploy/xornet.py).

2. Dump the file with test cases attached to the model.

   ```
   python3 dump_with_testcase_mge.py xornet_deploy.mge -o xornet.mge -d "#rand(0.1, 0.8, 4, 2)"
   ```

3. Verify the correctness by running `load_and_run` at the target platform.

   ```
   load_and_run xornet.mge
   ```
