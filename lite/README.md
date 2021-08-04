# Lite

It is a lite warper of MegEngine, to enable MegEngine easy to be integrated in 
user's SDK

## bazel build 

目前支持内部 bazel 和 CMake 编译，支持 C++/C, Python 接口，
下面是 bazel 中 lite_shared 目标的编译，可以作为其他目标的编译的参考，
该编译依赖内部 bazel 编译以及 megvii3。

### 配置编译环境

需要使用 megvii3 workspace 来完成 bazel 的编译

#### Clone megvii3 安装 bazel

```bash
    git clone git@git-core.megvii-inc.com:brain-sdk/megvii3.git
    ./utils/bazel/get_bazel.sh
```

#### Clone megbrain
```
    git submodule update brain/megbrain brain/midout
```

### 编译 x86 CUDA 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared --cpu="k8" \
        --compiler="gcc7_cuda10" -c opt
```

### 编译 x86 CPU 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared --cpu="k8" \
        --compiler="gcc9" -c opt
```

### 编译 arm OpenCL 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared_shared --cpu=android_aarch64 \
        -c opt --define enable_opencl=1  --define enable_opencl_search=1
```
### 编译 arm opencl lite_examples
bazel-3.0.0-megvii2 build //brain/megbrain/lite:lite_shared_examples \
--cpu=android_aarch64 --define enable_opencl=1  --define enable_opencl_search=1
####如何运行snpe_loder 的lite_exampes 请查看下面的wiki
https://wiki.megvii-inc.com/pages/viewpage.action?pageId=268786906

### 编译 armv7 CPU 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared --cpu=android_armv7 \
        -c opt
```

### 编译 arm64 CPU 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared --cpu=android_aarch64 \
        -c opt
```

### 编译 arm64 CPU v8.2 版本

```bash
    ./bazel build //brain/megbrain/lite:lite_shared --cpu=android_aarch64 \
       --copt -march=armv8.2-a+fp16+dotprod  -c opt
```

## 同时支持cmake构建
cmake构建参考scripts/cmake-build/BUILD_README.md,下面example表示同时支持编译megengine
和RKNPU后端且打开OpenCL的release模式
```bash
EXTRA_CMAKE_ARGS="-DANDROID_NATIVE_API_LEVEL=24 -DLITE_BUILD_WITH_RKNPU=ON -DMGE_WITH_OPENCL=ON \
-DMGE_OPENCL_SEARCH_ALGO=ON -DCUSTOM_C_OPR_INIT_FUNC=custom_loader_func" ./scripts/cmake-build/cross_build_android_arm_inference.sh"
```
* 如果需要支持性能分析的 profile 功能，则需要在编译时候加上
 --copt -DMGB_ENABLE_JSON=1 该参数
* 如果需要支持 fast-run 功能则需要加上
 --copt -DMGB_ENABLE_FASTRUN=1，开启 fast-run 功能
* 如果编译 arm64，可以加上 --copt -mcpu=cortex-a53 选项进行优化。

### midout 裁减编译
具体 midout 的裁减原理见 megbrain 中 midout 裁减，裁减方法见 MegBrain 
和 MegEngine 的裁减方法

## 模型

### 支持的模型

lite 目前支持只支持 MegEngine dump 的模型格式，可以加载的模型文件包括原始
的模型文件，原始的加密模型，pack 之后的加密或者非加密模型。加密算法以及
加密的秘钥可以用户自定义，然后注册到 lite 中，详见 example 中加解密部分。

* 原始模型未加密：直接将完成训练的模型在 MegEngine 环境中进行 dump 生成的模型
* 原始加密模型：将上述 dump 的模型通过加密算法进行加密，lite 提供两种默认
的加密算法，在 tools 中，分别为 aes 和 rc4. 对应为：aes_encypt.sh 和
rc4_encrypt.cpp，rc4_encrypt.cpp 需要编译生成可执行文件。这种方式加密的模型在
加载时候需要在 Config 中配置模型的加密方式。
* pack 之后的模型：模型结构将在下面介绍，可以将上面加密或者未加密的模型，和下面
定义的 json config 文件一同打包为一个 pack 之后的模型，可以使用 tools 下面
的 pack_model_and_info.py 工具中完成，pack_model_and_info.py 的使用详见其中
的 help 输出。

### 模型结构

不同的模型文件主要是通过 pack 之后的模型文件中的 model_tag 来区分.

* 打包处理之后的文件：
  模型打包过程可以通过脚本 pack_model_and_json.py 来完成，其将模型info文件（
  可以是任意格式，推荐使用JSON，可以加密也可以不加密）和加密或者未加密的模型文件
  一同打包在一起，并在文件开头加上 Header 来帮助解析。
* 原始文件和原始的加密文件没有 Header 和模型 info部分，模型加载需要的信息
  可以通过 Config 和 NetworkIO 进行传递。

### Header

Header 部分最开始为一个明文固定model_tag，目前定义为"packed_model"字符串，
后面主要包含模型文件各个部分的信息，每个部分的加密方式，load 模型时候可以
调用相应的解密方法对各个部分进行解密，以及model infomation 部分的解析方法。
具体细节参考lite/src/parse_model/pack_model.fbs

### Info部分

Info 部分主要用来解释模型，如用户关心的：模型的输入数据的格式，模型运行的平台
等信息，这部分信息也可以用于用户进行 check 运行的模型是否在指定的条件下运行。
由于这个 Info 部分不同的用户需求不一致，想传递的信息也无法统一，所以目前
Lite 中提供自定义的方式，用户可以自定义自己 Info 部分的类容，并在 Header 中
指定 **Info 解析方式名字** ，并注册以该名字为 key 的解析函数到 Lite 中，
以这样方式来可以实现用户自定义 Info 格式。同时，Lite 中也提供了一套定义好的
格式，其名字为 "LITE_default"，并已经实现了对应的解析函数，该 info
为 JSON 格式，具体内容定义如下：

```json
{
    "name": "shufflenet_test",
    "valid": true,
    "version": "8.9999.0",
    "has_compression": false,
    "device": {
        "type": "CPU",
        "device_id": 0,
        "number_threads": 1,
        "use_tensor_rt": false,
        "enable_inplace_model": false
    },
    "options":{
        "weight_preprocess": false,
        "var_sanity_check_first_run": true,
        "const_shape": false,
        "jit_level": 0,
        "record_level": 0
    },
    "IO":{
        "inputs":[
             {
                "name": "data",
                "io_type": "value",
                "is_host": true,
                "dtype": "float32",
                "shape": {
                    "dim0": 1,
                    "dim1": 3,
                    "dim2": 224,
                    "dim3": 224
                }
            }
        ],
        "outputs":[
             {
                "name": "TRUE_DIV(EXP[12065],reduce0[12067])[12077]",
                "io_type": "value",
                "is_host": true,
                "dtype": "float32",
                "shape": {
                    "dim0": 1,
                    "dim1": 1000,
                    "dim2": 0,
                    "dim3": 0
                }
            }
        ]
    }
}
```

* model_name: 指这个模型的名字，用户可以用来验证是否运行了正确的模型，
和 Header 部分中的进行对比 check
* valid: 指在这个 info 文件中的设置是否影响模型的 Config
* version: 指模型对应的 megbrain 的版本号，load 模型时候会进行 check
* has_compression: 标识这个模型文件中 tensor 的数据是否压缩过
* device: 目前支持字段包括："CPU","CUDA","OPENCL","ATLAS"
* number_threads 和 is_inplace_model : 只有在 device 为 CPU 的情况下才生效
* IO::inputs::type: 包括 value,shape，详见 include"network.h"
* IO::inputs::is_host: 值输入数据来自 device 或者来自 host 端
* IO::outputs::is_host: 值输出数据将保存在 device 或者 host 端
* IO::outputs::shape::dimx: 如果为0，则便是该 dim 无效

### Model部分

可以是加密的模型文件或者未加密的模型文件

## 使用

丰富的使用方法详见文件 example 中文档和对应的 example。

## 工具

目前 lite 中有三个工具保存在 tools 目录中，其他 megbrain 工具
没有包含在内，分别为：

* pack_model_and_info.py 为上面提到的模型打包工具，其为一个
  python 脚本，可以直接用其对已有的模型和模型 information 的文件，按照上面
  的格式进行打包模型，用户可以指定模型名字，模型加密方式，模型信息
  文件加密方式，解析方式等，如下：

    ```bash
    python3 pack_model_and_info.py --input-model xxx.mge \
        --model-name="shufflenet_test" \
        --model-cryption="RC4_default" \
        --input-info xxx.json \
        --info-cryption="RC4_default" \
        --info-parser="LITE_default" \
        -o xxx.lite
    ```
* aes_encrypt.sh 为一个 aes 加密方式的加密脚本，可以将一个文件，
通过指定的的 key 加密成一个 aes 加密的文件，其中 key 为 32 个字节
16进制数。
    ```bash
    aes_encrypt.sh  xxx.mdl  xxx_encrypted.mdl \
        000102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F
    ```

* rc4_encypt.cpp 可以被编译成为一个 rc4 加密的工具，这个工具可以通过
  制定的 key 或者默认的 key 加密制定的文件，支持 rc4 方法和
  simple_fast_rc4 两种方法，支持自定义 key。
    * bazel 编译 x86 命令为：
    ```bash
    bazel build //brain/megbrain/lite:rc4_encryptor \
        --cpu='k8' --compiler='gcc9'
    ```
    * 加密文件，具体用法见 help
    ```bash
    rc4_encryptor encrypt_predefined_rc4 \
        to_be_encrypt.file encrypted.file
    ```
