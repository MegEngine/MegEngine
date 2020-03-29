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

## `load_and_run --input` the dumped mge file

You can also use `--input` to set mge file's input, this argument support these 4 formats:

1. PPM/PGM image file.
      
      PPM/PGM is supported by OpenCV  and simple to parse, you can easily use `cv::imwrite` to generate one.

      ```
      load_and_run model.mge --input "data:image.ppm"
      ```
      
      `data` is blob name and `image.ppm` is file path, we use `:` to seperate key and value. Please note that `"` is necessary in terminal.

2. npy file.

      npy is `Numpy` file format, here is a Python example

      ```
      import numpy as np
      import cv2
      mat = cv2.imread('file.jpg')
      np.save('image.npy', mat)
      arr = np.array([[[1.1, 1.2],[100, 200.0]]], dtype=np.float32)
      np.save('bbox.npy', arr)
      ```

      then `load_and_run` the model

      ```
      load_and_run model.mge --input data:image.npy;bbox.npy
      ``` 

3. json format.

      For json format, you have to identify data type and blob shape. Here is a Python example

      ```
      import numpy as np
      import json
      import cv2
      bbox = np.array([[[1.1, 1.2],[100, 200.0]]], dtype=np.float32)
      obj = dict()
      obj['shape'] = bbox.shape
      obj['raw'] = bbox.flatten().tolist()
      obj['type'] = str(bbox.dtype)
      json_object = dict()
      json_object['bbox'] = obj
      json_str = json.dumps(json_object)
      with open('bbox.json', 'w') as f:
      f.write(json_str)
      f.flush()
      f.close()
      ```
      
      The json loader in `load_and_run` is not fully implement [RFC7159](https://tools.ietf.org/html/rfc7159), it does not support `boolean` and `utf` string format which is useless during inference.

      Now let's `load-and-run` the model with json file
      
      ```
      load_and_run model.mge --input data:image.npy:bbox:bbox.json
      ```

      Mutiple key-value pair could be seperated with `;`.

4. plain string format.

      Also, you can give the value directly
      
      ```
      load_and_run model.mge --input data:image.ppm --input "bbox:[0,0],[200.0,200.0]" --input "batchid:0"
      ```

      1. `bbox` shape is `[1,2,2]` for `[0,0],[200.0,200.0]`. In order to facilitate user experience, the string parser would add an extra axis for input, thus `bbox:0` is correspond to `[1]` and `bbox:[0]` means that the shape is `[1,1]`

      2. Since we can only identify `int32` and `float32` from this format, don't forget `.` for float number.
