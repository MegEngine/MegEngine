## LiteTest
LiteTest 是对 MegEngine Lite 的功能进行测试工具，其中包括 Tensor 的测试，NetWork的测试，以及一些基本功能的单元测试，另外其还支持设备模型兼容性测试。

### Network 测试
Network 测试主要在文件 test_network.cpp，test_network_c.cpp，test_network_options.cpp 文件中：
* test_network.cpp：测试 MegEngine Lite 的 C++ 接口是否正确，主要测试 Network 的基本功能，包括，模型加载，模型配置，模型运行，模型加解密，模型打包等功能。
* test_network_options.cpp：测试 MegEngine Lite 的 C++ 接口中配置模型的优化选项是否正确，比如 record，weight_preprocess 等。
* test_network_c.cpp：主要测试 MegEngine Lite 的纯 C 接口是否正确，里面完全调用 MegEngine Lite 的纯 C 接口进行 Network 的推理。

### Tensor 测试
Tensor 测试主要测试 MegEngine Lite 中 Tensor 的使用是否正确，是否满足设计需要，有两个文件，分别是 test_tensor.cpp, test_tensor_c.cpp 。

### 设备模型兼容性测试
除了基本的功能测试外，还支持设备模型兼容性测试，主要实现在 test_network.cpp 的IONoCopyRecordAx中。
#### 基本原理
* IONoCopyRecordAx test 读取指定目录下的所有模型文件，目前定义为：ax_models 文件夹。
* IONoCopyRecordAx test使用 MegEngine Lite 接口遍历上面读取到的模型文件。
* 配置各种 MegEngine Lite 的参数。
* 运行模型，将模型运行之后的结果和正确的结果进行对比。
* 统计兼容的模型，如果某个模型报错，或者计算结果不正确，则这个设备上不支持这个模型。
* 统计所有成功的模型和失败的模型，输出 log 。

用户可以在 test 目录下新建 resource/lite/ax_models 目录，并将将需要测试的模型放到该目录，运行这个 test 则可以完成测试。
