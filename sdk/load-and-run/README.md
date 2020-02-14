# Load and Run

Load a model and run, for testing/debugging/profiling.

## Build
*megvii3 build*
```sh
bazel build //brain/megbrain:load_and_run
```

See [mnist-example](../mnist-example) for detailed explanations on build.

## Dump Model

There are two methods to dump model:

1. Dump by `MegHair/utils/debug/load_network_and_run.py --dump-cpp-model
   /path/to/output`, to test on random inputs. Useful for profiling.
2. Pack model as specified by
   [`dump_with_testcase.py`](dump_with_testcase.py), and use
   that script to dump model. This is useful for checking correctness on
   different platforms.

### Input File for `dump_with_testcase.py`

The input file must be a python pickle. It can be in one of the following two
formats:

1. Contain a network that can be loaded by `meghair.utils.io.load_network`; in
   such case, `--data` must be given and network output evaulated on current
   computing device is used as groundtruth. All output vars would be checked.
   The input data can be one of the following:
   1. In the format `var0:file0;var1:file1...` meaning that `var0` should use
      image file `file0`, `var1` should use image `file1` and so on. If there
      is only one input var, the var name can be omitted. This can be combined
      with `--resize-input` option.
   2. In the format `var0:#rand(min, max, shape...);var1:#rand(min, max)...` 
      meaning to fill the corresponding input vars with uniform random numbers 
      in the range `[min, max)`, optionally overriding its shape.
2. Contain a dict in the format `{"outputs": [], "testcases": []}`, where
   `outputs` is a list of output `VarNode`s and `testcases` is a list of test
   cases. Each test case should be a dict that maps input var names to
   corresponding values as `numpy.ndarray`. The expected outputs should also be
   provided as inputs, and correctness should be checked by `AssertEqual`. You
   can find more details in `dump_with_testcase.py`.

### Input File for `dump_with_testcase_mge.py`

The input file is obtained by calling `megengine.jit.trace.dump()`.
`--data` must be given.

## Example

1. Obtain the model file by running [xornet.py](../../python_module/examples/xor/xornet.py)

2. Dump the file with test cases attached to the model.

    ```
    python3 dump_with_testcase_mge.py xornet_deploy.mge -o xornet.mge -d "#rand(0.1, 0.8, 4, 2)"
    ```

    The dumped file `xornet.mge` can be loaded by `load_and_run`.
