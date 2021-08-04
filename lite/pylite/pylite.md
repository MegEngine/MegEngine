# PyLite

Lite的python接口提供更加方便灵活的使用Lite进行模型Inference，支持各种平台上运行，X86-CUDA，X86-CPU，Arm-CPU，Arm-CUDA平台。

## 安装
### whl包安装
Lite python接口的whl包会随着megbrain的发版发布，版本号和megbrain保持一致，目前发布的Lite的whl包，包括Linux、windows、macos平台，这些平台可以直接通过pip3安装。
```shell
    python3 -m pip install --upgrade pip
    python3 -m pip install megenginelite -i  https://pypi.megvii-inc.com/simple
```
### develop 安装
开发模式下，可以使用Cmake编译出lite动态库liblite.so/liblite.dll/liblite_shared.dylib，并使用这个动态库进行开发和debug。该方式安装的pylite只能在本地机器上使用，不能copy到其他机器上使用。
* 编译liblite.so。使用cmake编译出liblite.so
    * clone megbrain工程到本地
    ```shell
    git clone git@git-core.megvii-inc.com:brain-sdk/MegBrain.git
    ```
    * 进行Cmake编译，这里的cmake编译同megbrain的cmake编译，使用参数和宏也完全一样
    * 编译准备
    ```shell
    cd MegBrain
    sh ./third_party/prepare.sh
    mkdir build
    cd build 
    ```
    * 编译X86-CUDA版本
    ```shell
    cmake .. -DMGE_WITH_CUDA=ON -DMGE_WITH_TEST=ON -DCMAKE_BUILD_TYPE=Release &&  make -j$(nproc)
    ```
    * 编译X86 CPU Only版本
    ```shell
    cmake .. -DMGE_WITH_CUDA=OFF -DMGE_WITH_TEST=ON -DCMAKE_BUILD_TYPE=Release &&  make -j$(nproc)
    ```
    * 编译完成之后，liblite.so 保存在build目录中的lite文件下
    * 将liblite.so copy到megenginelite的python源文件目录下，就可以使用megenginelite了。
    ```shell
    MegBrain的工程目录为 ${mgb_hone}
    cp ${mgb_hone}/build/lite/liblite.so ${mgb_home}/lite/pylite/megenginelite/
    cd ${mgb_home}/lite/pylite
    python3 -m "import megenginelite"
    ```
    这样就可以在${mgb_home}/lite/pylite 目录下面开发和debug lite的python接口了

## python3中使用megenginelite
Lite的python接口是对其C/C++接口的一层封装，他们使用的模型都是相同的模型格式。megenginelite提供两种数据接口，分别是LiteTensor和LiteNetwork。

### LiteTensor
LiteTensor提供了用户对数据的操作接口，提供了接口包括:
* fill_zero: 将tensor的内存设置为全0
* share_memory_with: 可以和其他LiteTensor的共享内存
* copy_from: 从其他LiteTensor中copy数据到自身内存中
* reshape: 改变该LiteTensor的shape，内存数据保持不变
* slice: 对该LiteTensor中的数据进行切片，需要分别指定每一维切片的start，end，和step。
* set_data_by_share: 调用之后使得该LiteTensor中的内存共享自输入的array的内存，输入的array必须是numpy的ndarray，并且tensor在CPU上
* set_data_by_copy: 该LiteTensor将会从输入的data中copy数据，data可以是list和numpy的ndarray，需要保证data的数据量不超过tensor的容量，tensor在CPU上
* to_numpy: 将该LiteTensor中数据copy到numpy的array中，返回给用户，如果是非连续的LiteTensor，如slice出来的，将copy到连续的numpy array中，该接口主要数为了debug，有性能问题。

#### 使用example
* LiteTensor 设置数据example
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
* tensor 共享内存example
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
更多的使用可以参考pylite中test/test_tensor.py中的使用
### LiteNetwork
LiteNetwork主要为用户提供模型载入，运行等功能。使用的模型见lite的readme中关于模型的部分
* CPU基本模型载入运行的example
```
def test_network_basic():
    source_dir = os.getenv("LITE_TEST_RESOUCE")
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
* CUDA上使用device内存作为模型输入，需要在构造network候配置config和IO信息
```
def test_network_device_IO():
    source_dir = os.getenv("LITE_TEST_RESOUCE")
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
更多的使用可以参考pylite中test/test_network.py和test/test_network_cuda.py中的使用
