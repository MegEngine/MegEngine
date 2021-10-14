# PyLite

Lite 的 python 接口提供更加方便灵活的使用 Lite 进行模型 Inference，满足如下条件的环境都可以使用:

* OS 可以安装 [Python3](https://www.python.org/downloads/)
* [BUILD_README](../../scripts/cmake-build/BUILD_README.md) 中支持推理编译的平台

## 安装
### whl 包安装
目前预编译发布的 Lite 的 whl 包详情如下:

* 提供 Linux-x64(with CUDA)、windows-x64(with CUDA)、macos-x64(cpu only) 平台预编译包
* 可以直接通过 pip3 安装。其他 OS-ARCH 的包，如有需要，可以 build from src 参考 [BUILD_README](../../scripts/cmake-build/BUILD_README.md)
* 预编译包的构建流程可以参考 [BUILD_PYTHON_WHL_README.md](../../scripts/whl/BUILD_PYTHON_WHL_README.md)

开源版本: 预编译的包会随着 MegEngine 的发版发布，版本号和 MegEngine 保持一致,安装方式:

```shell
python3 -m pip install --upgrade pip
python3 -m pip install megengine -f https://megengine.org.cn/whl/mge.html
```
安装后， 就可以通过 import megenginelite 进行使用了

### develop 调试

开发模式下，可以使用 Cmake 编译出 lite 动态库,依然参考 [BUILD_README](../../scripts/cmake-build/BUILD_README.md):

* Windows 平台，编译出来的 dll 是 lite_shared_whl.dll
* None Windows 平台，编译出来的 so 是 liblite_shared_whl.so

* 编译上述库的步骤:
    * clone 代码
    ```shell
    开源版本：git clone git@github.com:MegEngine/MegEngine.git
    ```
    * 编译准备
    ```shell
    开源版本: cd MegEngine
    bash ./third_party/prepare.sh
    ```
    * 编译 HOST 版本：
    ```shell
    ./scripts/cmake-build/host_build.sh
    ```
    * 编译 HOST 版本 (with CUDA):
    ```shell
    ./scripts/cmake-build/host_build.sh -c
    ```
    * 编译 Android 平台：

    ```shell
    scripts/cmake-build/cross_build_android_arm_inference.sh
    ```

    * 其他OS-ARCH可参考 [BUILD_README](../../scripts/cmake-build/BUILD_README.md)
    * 编译完成之后，相应的库可在 build_dir 下找到， 这里假设它的目录是LITE_LIB_PATH=path_of_lite_shared_whl
    * 开始使用 megenginelite
    ```shell
    export LITE_LIB_PATH=path_of_lite_shared_whl
    export PYTHONPATH=lite/pylite:$PYTHONPATH
    然后就可以 import megenginelite 进行使用了
    ```

## python3 中使用 megenginelite
Lite 的 python3 接口是对其 C/C++ 接口的一层封装，他们使用的模型都是相同的模型格式。megenginelite 提供两种数据接口，分别是 LiteTensor 和 LiteNetwork。

### LiteTensor
LiteTensor 提供了用户对数据的操作接口，提供了接口包括:
* fill_zero: 将 tensor 的内存设置为全0
* share_memory_with: 可以和其他 LiteTensor 的共享内存
* copy_from: 从其他 LiteTensor 中 copy 数据到自身内存中
* reshape: 改变该 LiteTensor 的 shape，内存数据保持不变
* slice: 对该 LiteTensor 中的数据进行切片，需要分别指定每一维切片的 start，end，和 step。
* set_data_by_share: 调用之后使得该 LiteTensor 中的内存共享自输入的 array 的内存，输入的 array 必须是numpy 的 ndarray，并且 tensor 在 CPU 上
* set_data_by_copy: 该 LiteTensor 将会从输入的 data 中 copy 数据，data 可以是 list 和 numpy 的 ndarray，需要保证 data 的数据量不超过 tensor 的容量，tensor 在 CPU 上
* to_numpy: 将该 LiteTensor 中数据 copy 到 numpy 的 array 中，返回给用户，如果是非连续的 LiteTensor，如 slice 出来的，将 copy 到连续的 numpy array 中，该接口主要数为了 debug，有性能问题。

#### 使用 example
* LiteTensor 设置数据 example
```
def test_tensor_set_data():
    layout = LiteLayout([2, 16], "int8")
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 2 * 16

    data = [i for i in range(32)]
    tensor.set_data_by_copy(data)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

    arr = np.ones([2, 16], "int8")
    tensor.set_data_by_copy(arr)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == 1

    for i in range(32):
        arr[i // 16][i % 16] = i
    tensor.set_data_by_share(arr)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

    arr[0][8] = 100
    arr[1][3] = 20
    real_data = tensor.to_numpy()
    assert real_data[0][8] == 100
    assert real_data[1][3] == 20
```
* tensor 共享内存 example
```python
def test_tensor_share_memory_with():
    layout = LiteLayout([4, 32], "int16")
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 4 * 32 * 2

    arr = np.ones([4, 32], "int16")
    for i in range(128):
        arr[i // 32][i % 32] = i
    tensor.set_data_by_share(arr)
    real_data = tensor.to_numpy()
    for i in range(128):
        assert real_data[i // 32][i % 32] == i

    tensor2 = LiteTensor(layout)
    tensor2.share_memory_with(tensor)
    real_data = tensor.to_numpy()
    real_data2 = tensor2.to_numpy()
    for i in range(128):
        assert real_data[i // 32][i % 32] == i
        assert real_data2[i // 32][i % 32] == i

    arr[1][18] = 5
    arr[3][7] = 345
    real_data = tensor2.to_numpy()
    assert real_data[1][18] == 5
    assert real_data[3][7] == 345
```
更多的使用可以参考 pylite 中 test/test_tensor.py 中的使用
### LiteNetwork
LiteNetwork 主要为用户提供模型载入，运行等功能。使用的模型见 lite 的 readme 中关于模型的部分
* CPU 基本模型载入运行的 example
```
def test_network_basic():
    source_dir = os.getenv("LITE_TEST_RESOURCE")
    input_data_path = os.path.join(source_dir, "input_data.npy")
    # read input to input_data
    input_data = np.load(input_data_path)
    model_path = os.path.join(source_dir, "shufflenet.mge")

    network = LiteNetwork()
    network.load(model_path)

    input_name = network.get_input_name(0)
    input_tensor = network.get_io_tensor(input_name)
    output_name = network.get_output_name(0)
    output_tensor = network.get_io_tensor(output_name)

    assert input_tensor.layout.shapes[0] == 1
    assert input_tensor.layout.shapes[1] == 3
    assert input_tensor.layout.shapes[2] == 224
    assert input_tensor.layout.shapes[3] == 224
    assert input_tensor.layout.data_type == LiteDataType.LITE_FLOAT
    assert input_tensor.layout.ndim == 4

    # copy input data to input_tensor of the network
    input_tensor.set_data_by_copy(input_data)
    for i in range(3):
        network.forward()
        network.wait()

    output_data = output_tensor.to_numpy()
    print('shufflenet output max={}, sum={}'.format(output_data.max(), output_data.sum()))
```
* CUDA 上使用 device 内存作为模型输入，需要在构造 network 候配置 config 和 IO 信息
```
def test_network_device_IO():
    source_dir = os.getenv("LITE_TEST_RESOURCE")
    input_data_path = os.path.join(source_dir, "input_data.npy")
    model_path = os.path.join(source_dir, "shufflenet.mge")
    # read input to input_data
    input_data = np.load(input_data_path)
    input_layout = LiteLayout([1, 3, 224, 224])
    host_input_data = LiteTensor(layout=input_layout)
    host_input_data.set_data_by_share(input_data)
    dev_input_data = LiteTensor(layout=input_layout, device_type=LiteDeviceType.LITE_CUDA)
    dev_input_data.copy_from(host_input_data)

    # construct LiteOption
    options = LiteOptions()
    options.weight_preprocess = 1
    options.var_sanity_check_first_run = 0
    net_config = LiteConfig(device_type=LiteDeviceType.LITE_CUDA, option=options)

    # constuct LiteIO, is_host=False means the input tensor will use device memory
    input_io = LiteIO("data", is_host=False)
    ios = LiteNetworkIO()
    ios.add_input(input_io)

    network = LiteNetwork(config=net_config, io=ios)
    network.load(model_path)

    input_name = network.get_input_name(0)
    dev_input_tensor = network.get_io_tensor(input_name)
    output_name = network.get_output_name(0)
    output_tensor = network.get_io_tensor(output_name)

    # copy input data to input_tensor of the network
    dev_input_tensor.share_memory_with(dev_input_data)
    for i in range(3):
        network.forward()
        network.wait()

    output_data = output_tensor.to_numpy()
    print('shufflenet output max={}, sum={}'.format(output_data.max(), output_data.sum()))
```
更多的使用可以参考 pylite 中 test/test_network.py 和 test/test_network_cuda.py 中的使用
