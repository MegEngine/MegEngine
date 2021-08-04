# Example

在该 example 目录中实现了一系列调用 lite 接口来实现 inference 的例子，主要
是演示 lite 中不同接口的调用来实现不同情况下的 inference 功能。这里所有的 example 
都是使用 shufflenet 来进行演示。

## Example bazel 的编译和运行

* 参考主目录下面的 README.md 搭建 megvii3 bazel 的编译环境，编译 CPU 版本
```bash
    ./bazel build //brain/megbrain/lite:lite_examples --cpu="k8" \
        --compiler="gcc9" -c opt
```
* 运行时需要指定运行的具体 example 名字，运行的模型，模型运行的数据
 * 获取所有的 example 名字
```
    bazel-bin/brain/megbrain/lite/lite_examples
```
 * 运行 example，下面命令运行 basic_load_from_memory
```
    bazel-bin/brain/megbrain/lite/lite_examples \
        basic_load_from_memory \
        path-to-megbrain/lite/test/resource/lite/shufflenet.mge \
        path-to-megbrain/lite/test/resource/lite/input_data.npy
```

## basic 使用

* **实现在文件 basic.cpp 中, 包括 basic_load_from_path 和
 basic_load_from_memory**

* 该 example 使用 lite 来完成基本的 inference 功能，load 模型使用默认的配置，
进行 forward 之前将输入数据 copy 到输入 tensor 中，完成 forward 之后，再将
数据从输出 tensor 中 copy 到用户的内存中，输入 tensor 和输出 tensor 都是从
Network 中通过 name 来获取的，输入输出 tensor 的 layout 也可以从对应的 tensor
中直接获取获取，**输出 tensor 的 layout 必须在 forward 完成之后获取才是正确的。**

## 输入输出指定的内存

* **实现在 reset_io.cpp 中，包括两个 example，reset_input 和 reset_input_output
两个 example。**

* 该 example 中演示输入 tensor 的内存为用户指定的内存（该内存中已经保存好输入
数据），输出 tensor 也可以是用户指定的内存，这样 Network 完成 Forward 之后就会将数据
保存在指定的输出内存中。如此减少不必要的 memory copy 的操作。

* 主要是通过 tensor 中的 reset 接口，该接口可以重新指定 tensor 的内存和对应的
layout，如果 layout 没有指定，默认为 tensor 中原来的 layout。

* **该方法中由于内存是用户申请，需要用户提前知道输入，输出 tensor 对应的 layout，然后
根据 layout 来申请内存，另外通过 reset 设置到 tensor 中的内存，生命周期不由 tensor
管理，由外部用户来管理。**

## 输入输出指定 device 上内存

* **实现在 device_io.cpp 中，device_input 和 device_input_output 两个 example。**

* 该 example 中配置模型运行在 device(CUDA) 上，并且使用用户提前申请的 device 上的内存
作为模型运行的输入和输出。需要在 Network 构建的时候指定输入输出的在 device 上，不设置默认
在 CPU 上，其他地方和**输入输出为用户指定的内存**的使用相同

* 可以通过 tensor 的 is_host() 接口来判断该 tensor 在 device 端还是 host 端

## 申请 pinned host 内存作为输入

* **实现在 device_io.cpp 中，函数名字为 pinned_host_input。**

* 这个 example 中模型运行在 device(CUDA) 上，但是输入输出在 CPU 上，为了加速 host2device 的
copy，将 CPU 上的 input tensor 的内存指定提前申请为 cuda pinned 内存。目前如果输出
output tensor 不是 device 上的时候，默认就是 pinned host 的。

* 申请 pinned host 内存的方法是：构建 tensor 的时候指定 device，layout，以及 is_host_pinned
参数，这样申请的内存就是 pinned host 的内存。

    ```C
     bool is_pinned_host = true;
     auto tensor_pinned_input =
             Tensor(LiteDeviceType::LITE_CUDA, input_layout, is_pinned_host);
    ```

## 用户指定内存分配器

* **实现在 user_allocator.cpp 中，函数名为：config_user_allocator。**

* 这个例子中使用用户自定义的 CPU 内存分配器演示了用户设置自定义的 Allocator 的方法，用户自定义
内存分配器需要继承自 lite 中的 Allocator 基类，并实现 allocate 和 free 两个接口。目前在 CPU
上验证是正确的，其他设备上有待测试。

* 设置自定定义内存分配器的接口为 Network 中如下接口：
    ```C
    Network& set_memory_allocator(std::shared_ptr<Allocator> user_allocator);
    ```

## 多个 Network 共享同一份模型 weights

* **实现在 network_share_weights.cpp 中，函数名为：network_share_same_weights。**

* 很多情况用户希望多个 Network 共享同一份 weights，因为模型中 weights 是只读的，这样可以节省
模型的运行时内存使用量。这个例子主要演示了 lite 中如何实现这个功能，首先创建一个新的 Network，
用户可以指定新的 Config 和 NetworkIO 以及其他一些配置，使得新创建出来的 Network 完成不同的
功能。

* 通过已有的 NetWork load 一个新的 Network 的接口为 Network 中如下接口：
    ```C
        static void shared_weight_with_network(
            std::shared_ptr<Network> dst_network,
            const std::shared_ptr<Network> src_network);
    ```
    * dst_network: 指新 load 出来的 Network
    * src_network：已经 load 的老的 Network

## CPU 绑核

* **实现在 cpu_affinity.cpp 中，函数名为：cpu_affinity。**

* 该 example 之中指定模型运行在 CPU 多线程上，然后使用 Network 中的
set_runtime_thread_affinity 来设置绑核回调函数。该回调函数中会传递当前线程的 id 进来，用户可以
根据该 id 决定具体绑核行为，在多线程中，如果线程总数为 n，则 id 为 n-1 的线程为主线程。

## 用户注册自定义解密算法和 key

* **实现在 user_cryption.cpp 中，函数名为：register_cryption_method 和 update_aes_key 。**

* 这两个 example 主要使用 lite 自定义解密算法和更新解密算法的接口，实现了使用用户自定的解密算法
实现模型的 load 操作。在这个 example 中，自定义了一个解密方法，(其实没有做任何事情，
将模型两次异或上 key 之后返回，等于将原始模型直接返回)，然后将其注册到 lite 中，后面创建 Network 时候在其
config 中的 bare_model_cryption_name 指定具体的解密算法名字。在第二个 example 展示了对其
key 的更新操作。
目前 lite 里面定义好了几种解密算法：
    * AES_default : 其 key 是由 32 个 unsighed char 组成，默认为0到31
    * RC4_default : 其 key 由 hash key 和 enc_key 组成的8个 unsigned char，hash
      key 在前，enc_key 在后。
    * SIMPLE_FAST_RC4_default : 其 key 组成同 RC4_default。
大概命名规则为：前面大写是具体算法的名字，'_'后面的小写，代表解密 key。
具体的接口为：
    ```C
    bool register_decryption_and_key(std::string decrypt_name,
                                    const DecryptionFunc& func,
                                    const std::vector<uint8_t>& key);
    bool update_decryption_or_key(std::string decrypt_name,
                                    const DecryptionFunc& func,
                                    const std::vector<uint8_t>& key);
    ```
register 接口中必须要求三个参数都是正确的值，update中 decrypt_nam 必须为已有的解密算法，
将使用 func 和 key 中不为空的部分对 decrypt_nam 解密算法进行更新

## 异步执行模式

* **实现在 basic.cpp 中，函数名为：async_forward。**

* 用户通过接口注册异步回调函数将设置 Network 的 Forward 模式为异步执行模式，
目前异步执行模式只有在 CPU 和 CUDA 10.0 以上才支持，在 inference 时异步模式，
主线程可以在工作线程正在执行计算的同时做一些其他的运算，避免长时间等待，但是
在一些单核处理器上没有收益。

## 纯 C example

* **实现在 lite_c_interface.cpp，函数名为：basic_c_interface，
device_io_c_interface，async_c_interface**

* Lite 完成对 C++ 接口的封装，对外暴露了纯 C 的接口，用户如果不是源码依赖 Lite
的情况下，应该使用纯 C 接口来完成集成。
* 纯 C 的所有接口都是返回一个 int，如果这个 int 的数值不为 0，则又错误产生，需要
调用 LITE_get_last_error 来获取错误信息。
* 纯 C 的所有 get 函数都需要先定义一个对应的对象，然后将该对象的指针传递进接口，
Lite 会将结果写入到 对应指针的地址里面。
