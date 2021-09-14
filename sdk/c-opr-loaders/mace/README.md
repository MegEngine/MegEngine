# Extern-C-Opr with MACE

### Build MegEngine `load_and_run` for arm64-v8a
NOTICE: build depends on [NDK](https://developer.android.com/ndk/downloads)
after download, please config env by:
```bash
export NDK_ROOT=path/to/ndk
export ANDROID_NDK_HOME=${NDK_ROOT}
export PATH=${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/:$PATH
```

```bash
cd $MEGENGINE_HOME
git checkout v1.0.0    (we only test v1.0.0 version)
./scripts/cmake-build/cross_build_android_arm_inference.sh -a arm64-v8a -r
```

After successfully built:
* load_and_run should be in `$MEGENGINE_HOME/build_dir/android/arm64-v8a/Release/install/bin`
* libmegengine.so should be in `$MEGENGINE_HOME/build_dir/android/arm64-v8a/Release/install/lib`

### Build MACE libraries for arm64-v8a with GPU runtime

```bash
cd $MACE_HOME
RUNTIME=GPU bash tools/cmake/cmake-build-arm64-v8a.sh
export SDKPATH=${MACE_HOME}/build/cmake-build/arm64-v8a/install
```
After successfully libmace.so should be in `$MACE_HOME/build/cmake-build/arm64-v8a/install/lib/libmace.so`

### Build MACE loader for MegEngine

If `SDKPATH` is not set, by default it's `./arm64-v8a`

You can run with debug mode(by adding `DEBUG=1` to make command), which will show more running information

### Prepare a MACE model(for example: resnet_50), wrap it with MegEngine extern c opr

```
python3 dump_model.py --input path/to/resnet_50.pb --param path/to/resnet_50.data --output resnet_50.mdl --config path/to/resnet_50.yml
```

`*.pb` file denotes the model structure, `*.data` denotes the model parameters

Check [here](https://github.com/XiaoMi/mace-models) to learn how to write yml files for MACE

### Run with load-and-run

First of all, send all files to the executed device(for example: /data/local/tmp/test/):

- load_and_run
- resnet_50.mdl
- libmace_loader.so
- libmegengine.so
- libmace.so

As mace build with `c++_shared` by default, but old AOSP device do not have `libc++_shared.so` by default, if you use this class devices
also need send it to devices, which always can be found at `${NDK_ROOT}/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so`

```
login to device
cd /path/to/ (for example: /data/local/tmp/test/)

MGB_MACE_RUNTIME=GPU MGB_MACE_OPENCL_CACHE_PATH=./ MGB_MACE_LOADER_FORMAT=NCHW LD_LIBRARY_PATH=. ./load_and_run resnet_50.mdl  --c-opr-lib libmace_loader.so  --input input-bs1.npy
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
