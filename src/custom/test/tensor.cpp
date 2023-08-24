#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP

#include "gtest/gtest.h"
#include "megbrain/comp_node.h"
#include "megbrain/custom/adaptor.h"
#include "megbrain/custom/tensor.h"
#include "megbrain/tensor.h"
#include "megbrain_build_config.h"

#define TENSOR_TEST_LOG 0

using namespace mgb;

namespace custom {

TEST(TestDevice, TestDevice) {
#if MGB_CUDA
    ASSERT_TRUE(Device::is_legal("x86"));
    ASSERT_TRUE(Device::is_legal(DeviceEnum::cuda));
    ASSERT_FALSE(Device::is_legal("cpu"));

    Device dev1;
    ASSERT_TRUE(dev1.str() == "invalid");

    dev1 = "x86";
    ASSERT_TRUE("x86" == dev1);

    Device dev2 = "cuda";
    ASSERT_TRUE(dev2 == "cuda");
    ASSERT_FALSE(dev2 == dev1);

    Device dev3 = dev2;
    ASSERT_TRUE(dev3 == dev2);
    ASSERT_FALSE(dev3 == dev1);

    Device dev4 = DeviceEnum::cuda;
    ASSERT_TRUE(dev4.enumv() == DeviceEnum::cuda);

#if TENSOR_TEST_LOG
    std::cout << dev1.str() << "\n"
              << dev2.str() << "\n"
              << dev3.str() << "\n"
              << dev4.str() << std::endl;
#endif

    CompNode compnode = to_builtin<CompNode, Device>(dev3);
    ASSERT_TRUE(compnode.to_string_logical() == "gpux:0");
    compnode = CompNode::load("cpu0:0");
    Device dev5 = to_custom<CompNode, Device>(compnode);
    ASSERT_TRUE(dev5.str() == "x86");

    std::vector<Device> devs1 = {"x86", "cuda", "x86"};
    megdnn::SmallVector<CompNode> compnodes = to_builtin<CompNode, Device>(devs1);
    ASSERT_TRUE(compnodes[0].to_string_logical() == "cpux:0");
    ASSERT_TRUE(compnodes[1].to_string_logical() == "gpux:0");
    ASSERT_TRUE(compnodes[2].to_string_logical() == "cpux:0");

    std::vector<Device> devs2 = to_custom<CompNode, Device>(compnodes);
    ASSERT_TRUE(devs2[0] == "x86");
    ASSERT_TRUE(devs2[1].str() == "cuda");
    ASSERT_TRUE(devs2[2] == "x86");
#endif
}

TEST(TestShape, TestShape) {
    Shape shape1, shape2;
    ASSERT_TRUE(shape1.ndim() == 0);

    shape1 = {16, 32, 8, 8};
    shape2 = shape1;
    ASSERT_TRUE(shape2.ndim() == 4);
    ASSERT_TRUE(shape2[0] == 16);
    ASSERT_TRUE(shape2[1] == 32);
    ASSERT_TRUE(shape2[2] == 8);
    ASSERT_TRUE(shape2[3] == 8);

    Shape shape3 = {16, 32, 8, 8};
    const Shape shape4 = shape1;
    ASSERT_TRUE(shape3 == shape4);
    shape3[0] = 32;
    ASSERT_FALSE(shape3 == shape4);
    ASSERT_TRUE(shape3[0] == 32);
    ASSERT_TRUE(shape4[0] == 16);

    Shape shape5 = {2, 3, 4};
    TensorShape bshape1 = to_builtin<TensorShape, Shape>(shape5);
    ASSERT_TRUE(bshape1.ndim == 3);
    ASSERT_TRUE(bshape1[0] == 2);
    ASSERT_TRUE(bshape1[1] == 3);
    ASSERT_TRUE(bshape1[2] == 4);
    bshape1 = {4, 2, 3};
    Shape shape6 = to_custom<TensorShape, Shape>(bshape1);
    ASSERT_TRUE(shape6.ndim() == 3);
    ASSERT_TRUE(shape6[0] == 4);
    ASSERT_TRUE(shape6[1] == 2);
    ASSERT_TRUE(shape6[2] == 3);

    Shape shape7;
    shape7.ndim(3);
    shape7[1] = 4;
    ASSERT_TRUE(shape7 == Shape({0, 4, 0}));

    std::vector<Shape> shapes1 = {{2, 3, 4}, {6}, {5, 7}};
    megdnn::SmallVector<TensorShape> bshapes = to_builtin<TensorShape, Shape>(shapes1);
    ASSERT_TRUE(bshapes[0].total_nr_elems() == 2 * 3 * 4);
    ASSERT_TRUE(bshapes[1].total_nr_elems() == 6);
    ASSERT_TRUE(bshapes[2].total_nr_elems() == 35);

    std::vector<Shape> shapes2 = to_custom<TensorShape, Shape>(bshapes);
    ASSERT_TRUE(shapes2[0] == Shape({2, 3, 4}));
    ASSERT_TRUE(shapes2[1] == Shape({6}));
    ASSERT_TRUE(shapes2[2] == Shape({5, 7}));
}

TEST(TestDType, TestDType) {
#if !MEGDNN_DISABLE_FLOAT16
    ASSERT_TRUE(DType::is_legal("uint8"));
    ASSERT_TRUE(DType::is_legal(DTypeEnum::bfloat16));

    DType dtype1, dtype2;
    ASSERT_TRUE(dtype1.str() == "invalid");

    dtype1 = "float32";
    ASSERT_TRUE(dtype1.str() == "float32");

    dtype2 = dtype1;
    DType dtype3 = dtype2;
    ASSERT_TRUE(dtype3 == dtype1);
    ASSERT_TRUE(dtype3 == "float32");

    dtype3 = "int8";
    ASSERT_FALSE("float32" == dtype3.str());
    ASSERT_FALSE(dtype3 == dtype2);

    DType dtype4 = DTypeEnum::int8, dtype5 = dtype3;
    ASSERT_TRUE(dtype4 == dtype5);
    ASSERT_TRUE(dtype4.is_compatible<int8_t>());
    ASSERT_FALSE(dtype4.is_compatible<uint8_t>());

    DType dtype6 = "int32";
    megdnn::DType bdtype1 = to_builtin<megdnn::DType, DType>(dtype6);
    ASSERT_TRUE(bdtype1.name() == std::string("Int32"));
    bdtype1 = megdnn::DType::from_enum(megdnn::DTypeEnum::BFloat16);
    DType dtype7 = to_custom<megdnn::DType, DType>(bdtype1);
    ASSERT_TRUE(dtype7.enumv() == DTypeEnum::bfloat16);

    std::vector<DType> dtypes1 = {"int8", "uint8", "float16"};
    megdnn::SmallVector<megdnn::DType> bdtypes =
            to_builtin<megdnn::DType, DType>(dtypes1);
    ASSERT_TRUE(bdtypes[0].name() == std::string("Int8"));
    ASSERT_TRUE(bdtypes[1].name() == std::string("Uint8"));
    ASSERT_TRUE(bdtypes[2].name() == std::string("Float16"));

    std::vector<DType> dtypes2 = to_custom<megdnn::DType, DType>(bdtypes);
    ASSERT_TRUE(dtypes2[0] == "int8");
    ASSERT_TRUE(dtypes2[1] == "uint8");
    ASSERT_TRUE(dtypes2[2] == "float16");
#endif
}

TEST(TestDType, TestDTypeQuantized) {
    DType quint8_1("quint8", 3.2, 15);
    DType quint8_2("quint8", 3.2, 15);
    DType quint8_3("quint8", 3.2, 16);
    DType quint8_4("quint8", 3.1, 15);

    ASSERT_TRUE(quint8_1 == quint8_2);
    ASSERT_FALSE(quint8_1 == quint8_3);
    ASSERT_FALSE(quint8_1 == quint8_4);

    ASSERT_TRUE(quint8_1.scale() == 3.2f);
    ASSERT_TRUE(quint8_1.zero_point() == 15);

    DType qint8("qint8", 3.3f);
    DType qint16("qint16", 3.4f);
    DType qint32("qint32", 3.5f);

    ASSERT_TRUE(qint8.scale() == 3.3f);
    ASSERT_TRUE(qint16.scale() == 3.4f);
    ASSERT_TRUE(qint32.scale() == 3.5f);

    ASSERT_TRUE(qint8.enumv() == DTypeEnum::qint8);
    ASSERT_TRUE(qint8.str() == "qint8");
}

TEST(TestFormat, TestFormat) {
    Format format1, format2("default");
    ASSERT_TRUE(format1.is_default());
    ASSERT_TRUE(format2.is_default());
    Format format3 = format1;
    ASSERT_TRUE(format3.is_default());
}

TEST(TestTensor, TestTensor) {
    CompNode builtin_device = CompNode::load("cpux:0");
    TensorShape builtin_shape = {3, 2, 4};
    megdnn::DType builtin_dtype = dtype::Int32{};

    DeviceTensorND dev_tensor(builtin_device, builtin_shape, builtin_dtype);
    Tensor tensor1 = to_custom<DeviceTensorND, Tensor>(dev_tensor);
    Tensor tensor2 = to_custom<DeviceTensorND, Tensor>(dev_tensor);
    Device device = tensor1.device();
    Shape shape = tensor1.shape();
    DType dtype = tensor1.dtype();

    ASSERT_TRUE(device == "x86");
    ASSERT_TRUE(shape.ndim() == 3);
    ASSERT_TRUE(shape[0] == 3);
    ASSERT_TRUE(shape[1] == 2);
    ASSERT_TRUE(shape[2] == 4);
    ASSERT_TRUE(shape == std::vector<size_t>({3, 2, 4}));
    ASSERT_TRUE(dtype == "int32");

    int* raw_ptr1 = tensor1.data<int>();
    for (size_t i = 0; i < tensor1.size(); i++)
        raw_ptr1[i] = i;

    int* raw_ptr2 = tensor2.data<int>();
    for (size_t i = 0; i < tensor2.size(); i++)
        ASSERT_TRUE(raw_ptr2[i] == static_cast<int>(i));

    Tensor tensor3 = tensor2;
    int* raw_ptr3 = tensor3.data<int>();
    for (size_t i = 0; i < tensor3.size(); i++)
        ASSERT_TRUE(raw_ptr3[i] == static_cast<int>(i));
    ASSERT_TRUE(raw_ptr1 == raw_ptr2);
    ASSERT_TRUE(raw_ptr1 == raw_ptr3);

    for (size_t i = 0; i < tensor3.size(); i++) {
        raw_ptr3[i] = -static_cast<int>(i);
    }
    for (size_t i = 0; i < tensor1.size(); i++) {
        ASSERT_TRUE(raw_ptr1[i] == -static_cast<int>(i));
    }

    DeviceTensorND new_dev_tensor = to_builtin<DeviceTensorND, Tensor>(tensor3);

    int* builtin_ptr = new_dev_tensor.ptr<int>();
    for (size_t i = 0; i < new_dev_tensor.shape().total_nr_elems(); i++) {
        ASSERT_TRUE(builtin_ptr[i] == -static_cast<int>(i));
    }
}

TEST(TestTensor, TestTensorQuantized) {
#if MGB_CUDA
    CompNode builtin_device = CompNode::load("gpux:0");
    TensorShape builtin_shape = {3, 2, 4};
    megdnn::DType builtin_dtype = dtype::Quantized8Asymm{3.2f, uint8_t(15)};

    DeviceTensorND dev_tensor(builtin_device, builtin_shape, builtin_dtype);

    Tensor tensor1 = to_custom<DeviceTensorND, Tensor>(dev_tensor);
    Tensor tensor2 = to_custom<DeviceTensorND, Tensor>(dev_tensor);
    Device device1 = tensor1.device(), device2 = tensor2.device();
    Shape shape1 = tensor1.shape(), shape2 = tensor2.shape();
    DType dtype1 = tensor1.dtype(), dtype2 = tensor2.dtype();

    ASSERT_TRUE(device1 == "cuda");
    ASSERT_TRUE(shape1.ndim() == 3);
    ASSERT_TRUE(shape1[0] == 3);
    ASSERT_TRUE(shape1[1] == 2);
    ASSERT_TRUE(shape1[2] == 4);
    ASSERT_TRUE(shape1 == std::vector<size_t>({3, 2, 4}));
    ASSERT_TRUE(dtype1 == "quint8");
    ASSERT_TRUE(dtype1.scale() == 3.2f);
    ASSERT_TRUE(dtype1.zero_point() == 15);

    ASSERT_TRUE(device1 == device2);
    ASSERT_TRUE(shape1 == shape2);
    ASSERT_TRUE(dtype1 == dtype2);
#endif
}

TEST(TestTensor, TestTensorAccessorND) {
    size_t N = 2, C = 4, H = 6, W = 8;
    CompNode builtin_device = CompNode::load("cpux");
    TensorShape builtin_shape = {N, C, H, W};
    megdnn::DType builtin_dtype = dtype::Int32{};

    DeviceTensorND dev_tensor(builtin_device, builtin_shape, builtin_dtype);
    int* builtin_ptr = dev_tensor.ptr<int>();
    for (size_t i = 0; i < dev_tensor.shape().total_nr_elems(); i++) {
        builtin_ptr[i] = i;
    }

    Tensor tensor = to_custom_tensor(dev_tensor);
    auto accessor = tensor.accessor<int32_t, 4>();
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    int32_t idx = n * C * H * W + c * H * W + h * W + w;
                    ASSERT_TRUE(accessor[n][c][h][w] == idx);
                }
            }
        }
    }
}

TEST(TestTensor, TestTensorAccessor1D) {
    CompNode builtin_device = CompNode::load("cpux");
    TensorShape builtin_shape = {32};
    megdnn::DType builtin_dtype = dtype::Float32{};

    DeviceTensorND dev_tensor(builtin_device, builtin_shape, builtin_dtype);
    float* builtin_ptr = dev_tensor.ptr<float>();
    for (size_t i = 0; i < dev_tensor.shape().total_nr_elems(); i++) {
        builtin_ptr[i] = i;
    }

    Tensor tensor = to_custom_tensor(dev_tensor);
    auto accessor = tensor.accessor<float, 1>();
    for (size_t n = 0; n < 32; ++n) {
        ASSERT_TRUE(accessor[n] == n);
    }
}

}  // namespace custom

#endif
