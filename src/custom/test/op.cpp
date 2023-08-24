#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP

#include "gtest/gtest.h"
#include "megbrain/comp_node.h"
#include "megbrain/custom/adaptor.h"
#include "megbrain/custom/op.h"
#include "megbrain/tensor.h"
#include "megbrain/test/helper.h"
#include "megbrain_build_config.h"

#define OP_TEST_LOG 0

using namespace mgb;

namespace custom {

TEST(TestCustomOp, TestCustomOpInfoSetter) {
    CustomOp test("TestOp", CUSTOM_OP_VERSION);
    test.set_description("Test Op")
            .add_input("lhs", "lhs of test op", {"float32", "int32"}, 2)
            .add_inputs(2)
            .add_input("rhs", "rhs of test op", {"float32", "int32"}, 2)
            .add_outputs(1)
            .add_output("out", "out of test op", {"float32", "int32"}, 2)
            .add_outputs(3);

    ASSERT_TRUE(test.op_type() == "TestOp");
    ASSERT_TRUE(test.op_desc() == "Test Op");
    ASSERT_TRUE(test.input_num() == 4);
    ASSERT_TRUE(test.output_num() == 5);

#if OP_TEST_LOG
    for (auto input : test.inputs_info()) {
        std::cout << input.str() << std::endl;
    }
    for (auto output : test.outputs_info()) {
        std::cout << output.str() << std::endl;
    }
#endif

    test.add_param("param1", "param1 - float", 1.23f)
            .add_param("param2", "param2 - float list", {2.34f, 3.45f})
            .add_param("param3", "param3 - string", "test-string")
            .add_param("param4", {"test", "string", "list"})
            .add_param("param5", 1);

#if OP_TEST_LOG
    ParamInfo pinfo = test.param_info();
    for (auto kv : pinfo.meta()) {
        std::cout << kv.str() << std::endl;
    }
#endif
}

void device_infer(
        const std::vector<Device>& inputs, const Param& params,
        std::vector<Device>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
    outputs[0] = inputs[1];
    outputs[1] = inputs[0];
}

void shape_infer(
        const std::vector<Shape>& inputs, const Param& params,
        std::vector<Shape>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
    outputs[0] = inputs[1];
    outputs[1] = inputs[0];
}

void dtype_infer(
        const std::vector<DType>& inputs, const Param& params,
        std::vector<DType>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
    outputs[0] = inputs[1];
    outputs[1] = inputs[0];
}

void format_infer(
        const std::vector<Format>& inputs, const Param& params,
        std::vector<Format>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
    outputs[0] = inputs[1];
    outputs[1] = inputs[0];
}

TEST(TestCustomOp, TestCustomOpFuncSetter) {
#if MGB_CUDA
    CustomOp test("TestOp", CUSTOM_OP_VERSION);
    test.set_description("Test Op Forward Backward Union")
            .add_input("lhs", "lhs of Test op", {"float32", "int32"}, 2)
            .add_input("rhs", "rhs of Test op", {"float32", "int32"}, 2)
            .add_output("outl", "outl of Test op", {"float32", "int32"}, 2)
            .add_output("outr", "outr of Test op", {"float32", "int32"}, 2)
            .add_param("scale_f", "scale_f", 1.f)
            .add_param("offset_i", "offset_i", 0)
            .add_param("device", "using for judge device", "x86");

    std::vector<Device> idevices = {"x86", "cuda"};
    std::vector<Shape> ishapes = {{2, 3}, {3, 4}};
    std::vector<DType> idtypes = {"int32", "float32"};
    std::vector<Format> iformats = {"default", "default"};
    Param param(test.param_info());

    std::vector<Device> odevices = test.infer_output_device(idevices, param);
    std::vector<Shape> oshapes = test.infer_output_shape(ishapes, param);
    std::vector<DType> odtypes = test.infer_output_dtype(idtypes, param);
    std::vector<Format> oformats = test.infer_output_format(iformats, param);

    ASSERT_TRUE(odevices.size() == 2);
    ASSERT_TRUE(oshapes.size() == 2);
    ASSERT_TRUE(odtypes.size() == 2);
    ASSERT_TRUE(oformats.size() == 2);

    ASSERT_TRUE(odevices[0] == "x86");
    ASSERT_TRUE(odevices[1] == "x86");
    ASSERT_TRUE(oshapes[0] == Shape({2, 3}));
    ASSERT_TRUE(oshapes[1] == Shape({2, 3}));
    ASSERT_TRUE(odtypes[0] == "int32");
    ASSERT_TRUE(odtypes[1] == "int32");
    ASSERT_TRUE(iformats[0].is_default());
    ASSERT_TRUE(iformats[1].is_default());

    test.set_device_infer(device_infer)
            .set_shape_infer(shape_infer)
            .set_dtype_infer(dtype_infer)
            .set_format_infer(format_infer);

    odevices = test.infer_output_device(idevices, param);
    oshapes = test.infer_output_shape(ishapes, param);
    odtypes = test.infer_output_dtype(idtypes, param);
    oformats = test.infer_output_format(iformats, param);

    ASSERT_TRUE(odevices.size() == 2);
    ASSERT_TRUE(oshapes.size() == 2);
    ASSERT_TRUE(odtypes.size() == 2);
    ASSERT_TRUE(oformats.size() == 2);

    ASSERT_TRUE(odevices[0] == "cuda");
    ASSERT_TRUE(odevices[1] == "x86");
    ASSERT_TRUE(oshapes[0] == Shape({3, 4}));
    ASSERT_TRUE(oshapes[1] == Shape({2, 3}));
    ASSERT_TRUE(odtypes[0] == "float32");
    ASSERT_TRUE(odtypes[1] == "int32");
    ASSERT_TRUE(iformats[0].is_default());
    ASSERT_TRUE(iformats[1].is_default());
#endif
}

void cpu_kernel(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    ASSERT_TRUE(inputs.size() == 2);
    ASSERT_TRUE(outputs.size() == 2);
    ASSERT_TRUE(params["device"] == "x86");
    ASSERT_TRUE(params["scale_f"] == 2.12f);
    ASSERT_TRUE(params["offset_i"] == 6);
    ASSERT_TRUE(inputs[0].shape() == Shape({3, 4}));
    ASSERT_TRUE(inputs[1].shape() == Shape({5, 6}));
    ASSERT_TRUE(outputs[0].shape() == Shape({5, 6}));
    ASSERT_TRUE(outputs[1].shape() == Shape({3, 4}));
    ASSERT_TRUE(inputs[0].device() == "x86");
    ASSERT_TRUE(inputs[1].device() == "x86");
    ASSERT_TRUE(outputs[0].device() == "x86");
    ASSERT_TRUE(outputs[1].device() == "x86");

    float scale_f = params["scale_f"].as<float>();
    int offset_i = params["offset_i"].as<int>();

    for (size_t i = 0; i < 5 * 6; ++i) {
        ASSERT_TRUE(inputs[1].data<float>()[i] == static_cast<float>(i));
        outputs[0].data<float>()[i] = inputs[1].data<float>()[i] * scale_f;
    }
    for (size_t i = 0; i < 3 * 4; ++i) {
        ASSERT_TRUE(inputs[0].data<int>()[i] == static_cast<int>(i));
        outputs[1].data<int>()[i] = inputs[0].data<int>()[i] + offset_i;
    }
}

TEST(TestCustomOp, TestCustomOpCompute) {
    std::shared_ptr<CustomOp> op =
            std::make_shared<CustomOp>("TestOp", CUSTOM_OP_VERSION);
    op->set_description("Test Op Forward Backward Union")
            .add_input("lhs", "lhs of Test op", {"float32", "int32"}, 2)
            .add_input("rhs", "rhs of Test op", {"float32", "int32"}, 2)
            .add_output("outl", "outl of Test op", {"float32", "int32"}, 2)
            .add_output("outr", "outr of Test op", {"float32", "int32"}, 2)
            .add_param("scale_f", "scale_f", 1.f)
            .add_param("offset_i", "offset_i", 0)
            .add_param("device", "using for judge device", "x86")
            .set_shape_infer(shape_infer)
            .set_dtype_infer(dtype_infer)
            .set_compute("x86", cpu_kernel);

    Param param(op->param_info());
    param["device"] = "x86";
    param["scale_f"] = 2.12f;
    param["offset_i"] = 6;

    HostTensorGenerator<dtype::Float32> gen_f;
    HostTensorGenerator<dtype::Int32> gen_i;
    auto host_i0 = gen_i({3, 4}), host_i1 = gen_f({5, 6});
    auto expect_o0 = gen_f({5, 6}), expect_o1 = gen_i({3, 4});
    for (size_t i = 0; i < 5 * 6; ++i) {
        host_i1->ptr<float>()[i] = static_cast<float>(i);
        expect_o0->ptr<float>()[i] = host_i1->ptr<float>()[i] * 2.12f;
    }
    for (size_t i = 0; i < 3 * 4; ++i) {
        host_i0->ptr<int>()[i] = static_cast<int>(i);
        expect_o1->ptr<int>()[i] = host_i0->ptr<int>()[i] + 6;
    }

    auto cn = CompNode::load("cpux");
    std::shared_ptr<SmallVector<DeviceTensorND>> x86_inps =
            std::make_shared<SmallVector<DeviceTensorND>>(2);
    x86_inps->at(0) = DeviceTensorND{cn};
    x86_inps->at(1) = DeviceTensorND{cn};
    x86_inps->at(0).copy_from(*host_i0).sync();
    x86_inps->at(1).copy_from(*host_i1).sync();

    std::shared_ptr<SmallVector<DeviceTensorND>> x86_oups =
            std::make_shared<SmallVector<DeviceTensorND>>(2);
    x86_oups->at(0) = DeviceTensorND{cn, {5, 6}, dtype::Float32{}};
    x86_oups->at(1) = DeviceTensorND{cn, {3, 4}, dtype::Int32{}};

    dispatch_custom_op(op, param, x86_inps, x86_oups);
    cn.sync();
    HostTensorND host_o0, host_o1;
    host_o0.copy_from(x86_oups->at(0)).sync();
    host_o1.copy_from(x86_oups->at(1)).sync();

    MGB_ASSERT_TENSOR_NEAR(*expect_o0, host_o0, 1e-6);
    MGB_ASSERT_TENSOR_NEAR(*expect_o1, host_o1, 1e-6);
}

}  // namespace custom

#endif
