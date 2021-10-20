/**
 * \file src/custom/test/op.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#if MGB_CUSTOM_OP

#include "gtest/gtest.h"
#include "megbrain/comp_node.h"
#include "megbrain/custom/data_adaptor.h"
#include "megbrain/custom/op.h"
#include "megbrain/tensor.h"
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

void cpu_kernel(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
#if OP_TEST_LOG
    std::cout << "Checking CPU Forward - " << params["device"].as<std::string>()
              << std::endl;
#endif
    ASSERT_TRUE(params["device"] == "x86");
}

void gpu_kernel(
        const std::vector<Tensor>& inputs, const Param& params,
        std::vector<Tensor>& outputs) {
    (void)inputs;
    (void)params;
    (void)outputs;
#if OP_TEST_LOG
    std::cout << "Checking GPU Forward - " << params["device"].as<std::string>()
              << std::endl;
#endif
    ASSERT_TRUE(params["device"] == "cuda");
}

TEST(TestCustomOp, TestCustomOpFuncSetter) {
#if MGB_CUDA
    CustomOp test("TestOp", CUSTOM_OP_VERSION);
    test.set_description("Test Op Forward Backward Union")
            .add_input("lhs", "lhs of Test op", {"float32", "int32"}, 2)
            .add_input("rhs", "rhs of Test op", {"float32", "int32"}, 2)
            .add_output("outl", "outl of Test op", {"float32", "int32"}, 2)
            .add_output("outr", "outr of Test op", {"float32", "int32"}, 2)
            .add_param("smooth", "smooth", 0.f)
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

    test.set_compute(cpu_kernel);
    DeviceTensorND cdev_itensor0(CompNode::load("cpux"), {3, 2}, dtype::Int32{});
    DeviceTensorND cdev_itensor1(CompNode::load("cpux"), {3, 2}, dtype::Float32{});
    DeviceTensorND cdev_otensor0(CompNode::load("cpux"), {3, 2}, dtype::Float32{});
    DeviceTensorND cdev_otensor1(CompNode::load("cpux"), {3, 2}, dtype::Int32{});

    std::vector<Tensor> cinputs = {
            to_custom_tensor(cdev_itensor0), to_custom_tensor(cdev_itensor1)};
    std::vector<Tensor> coutputs = {
            to_custom_tensor(cdev_otensor0), to_custom_tensor(cdev_otensor1)};
    param["device"] = "x86";
    test.compute(cinputs, param, coutputs);

    test.set_compute("cuda", gpu_kernel);
    DeviceTensorND gdev_itensor0(CompNode::load("gpux"), {3, 2}, dtype::Int32{});
    DeviceTensorND gdev_itensor1(CompNode::load("gpux"), {3, 2}, dtype::Float32{});
    DeviceTensorND gdev_otensor0(CompNode::load("gpux"), {3, 2}, dtype::Float32{});
    DeviceTensorND gdev_otensor1(CompNode::load("gpux"), {3, 2}, dtype::Int32{});

    std::vector<Tensor> ginputs = {
            to_custom_tensor(gdev_itensor0), to_custom_tensor(gdev_itensor1)};
    std::vector<Tensor> goutputs = {
            to_custom_tensor(gdev_otensor0), to_custom_tensor(gdev_otensor1)};
    param["device"] = "cuda";
    test.compute(ginputs, param, goutputs);
#endif
}

}  // namespace custom

#endif
