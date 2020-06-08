# Extern-C-Opr with MACE

### Build MegEngine `load_and_run` for arm64-v8a

```bash
cd $MEGENGINE_HOME
./scripts/cmake-build/cross_build_android_arm_inference.sh -a arm64-v8a
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
- opencl library(something like libOpenCL.so, libmali.so or libEGL.so ...) if you want to run it on GPU

```
RUNTIME=GPU OPENCPATH=/path/to/opencl DATAFORMAT=NCHW /path/to/load_and_run /path/to/resnet_50.mdl --c-opr-lib /path/to/libmace_loader.so
```

RUNTIME candidates:

- CPU
- GPU

Running with GPU runtime on android needs opencl library, one can set `OPENCLPATH` by using environment variable

We mainly use NCHW data format, if you have NHWC model, use environment `DATAFORMAT=NHWC`

if you want to run with HEXAGON runtime, more efforts should be made, please check [here](https://mace.readthedocs.io/en/latest/faq.html#why-is-mace-not-working-on-dsp).
