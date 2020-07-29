# Extern-C-Opr with MACE

### Build MegEngine `load_and_run` for arm64-v8a

```bash
cd $MEGENGINE_HOME
./scripts/cmake-build/cross_build_android_arm_inference.sh -a arm64-v8a -r
```

After successfully built, load_and_run should be in `$MEGENGINE_HOME/build_dir/android/arm64-v8a/Release/install/bin`

### Build MACE libraries for arm64-v8a with GPU runtime

```bash
cd $MACE_HOME
RUNTIME=GPU bash tools/cmake/cmake-build-arm64-v8a.sh

cp -r $MACE_HOME/build/cmake-build/arm64-v8a/install $MEGENGINE_HOME/sdk/c-opr-loaders/mace/arm64-v8a
```

### Build MACE loader for MegEngine

```
SDK_PATH=/path/to/mace-sdk make
```

If `SDK_PATH` is not set, by default it's `./arm64-v8a`

You can run with debug mode(by adding `DEBUG=1` to make command), which will show more running information

### Prepare a MACE model(for example: resnet_50), wrap it with MegEngine extern c opr

```
python3 dump_model.py path/to/resnet_50.pb path/to/resnet_50.data path/to/resnet_50.mdl path/to/resnet_50.yml
```

`*.pb` file denotes the model structure, `*.data` denotes the model parameters

Check [here](https://github.com/XiaoMi/mace-models) to learn how to write yml files for MACE

### Run with load-and-run

First of all, send all files to the executed device:

- load_and_run
- resnet_50.mdl
- libmace_loader.so

```
MGB_MACE_RUNTIME=GPU MGB_MACE_OPENCL_CACHE_PATH=/path/to/opencl MGB_MACE_LOADER_FORMAT=NCHW /path/to/load_and_run /path/to/resnet_50.mdl --c-opr-lib /path/to/libmace_loader.so
```

RUNTIME candidates:

- CPU
- GPU

`MGB_MACE_OPENCL_CACHE_PATH` is the directory path where OpenCL binary cache writes to (the cache file name is always `mace_cl_compiled_program.bin`), if the cache file does not exist then it will be created.

We mainly use NCHW data format, if you have NHWC model, use environment `MGB_MACE_LOADER_FORMAT=NHWC`

For CPU runtime, default running thread is 1, could be specified with `MGB_MACE_NR_THREADS=n`

if you want to run with HEXAGON runtime, more efforts should be made, please check [here](https://mace.readthedocs.io/en/latest/faq.html#why-is-mace-not-working-on-dsp).

### Tuning on specific OpenCL device

MACE supports tuning on specific SoC to optimize the performace on GPU, see [doc](https://mace.readthedocs.io/en/latest/user_guide/advanced_usage.html#tuning-for-specific-soc-s-gpu).

To enable this feature, use `MGB_MACE_TUNING_PARAM_PATH` env to give the path to the tuning param file.

To generate the tunig param file, give `MACE_TUNING=1` env and set the `MACE_RUN_PARAMETER_PATH` to the file name you want.

 ```bash
 # search for tuning param
 MACE_TUNING=1 MACE_RUN_PARAMETER_PATH=opencl/vgg16.tune_param MGB_MACE_RUNTIME=GPU MGB_MACE_OPENCL_PATH=opencl MGB_MACE_LOADER_FORMAT=NCHW ./load_and_run mace/vgg16.mdl --c-opr-lib libmace_loader.so --input 4d.npy

 # then run test using the param
 MGB_MACE_TUNING_PARAM_PATH=opencl/vgg16.tune_param MGB_MACE_RUNTIME=GPU MGB_MACE_OPENCL_PATH=opencl MGB_MACE_LOADER_FORMAT=NCHW ./load_and_run mace/vgg16.mdl --c-opr-lib libmace_loader.so --input 4d.npy
 ```