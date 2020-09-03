/**
 * \file src/cambricon/test/cambricon_runtime_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/helper.h"

#if MGB_CAMBRICON

#include "megbrain/cambricon/cambricon_runtime_opr.h"

using namespace mgb;

namespace {
class CnmlModelContext {
public:
    const CompNode& cn;
    bool batch_size_changable;
    cnmlModel_t model;
    cnmlTensor_t conv_input_tensor, relu_output_tensor;
    cnmlFusionOp_t fusion_op;
    bool built;
    CnmlModelContext(const CompNode& cn, bool batch_size_changable = false)
            : cn{cn},
              batch_size_changable{batch_size_changable},
              built{false} {}
    ~CnmlModelContext() {
        MGB_CNML_CHECK(cnmlDestroyTensor(&conv_input_tensor));
        MGB_CNML_CHECK(cnmlDestroyTensor(&relu_output_tensor));
        MGB_CNML_CHECK(cnmlDestroyFusionOp(&fusion_op));
        MGB_CNML_CHECK(cnmlDestroyModel(model));
    }

    void build() {
        auto&& cnrt_env = CompNodeEnv::from_comp_node(cn).cnrt_env();
        cnrt_env.activate();
        constexpr int core_num = 4;
        cnrtCoreVersion_t core_version = cnrt_env.device_info.core_version;

        // prepare parameter for addpad and conv
        constexpr int dim_num = 4;
        const int ni = 16, ci = 64, hi = 32, wi = 32;
        const int no = 16, co = 64, ho = 32, wo = 32;
        const int kh = 3, kw = 3;
        const int stride_h = 1, stride_w = 1, dilation = 1;
        const int pad_h = 2, pad_w = 2;

        // count tensor nums
        int conv_filter_count = co * kh * kw * ci;
        int conv_bias_count = 1 * 1 * 1 * co;

        // prepare cpu origin data
        std::vector<float> conv_filter_cpu_data(conv_filter_count);
        std::vector<float> conv_bias_cpu_data(conv_bias_count);

        // prepare input data for addpad
        unsigned int seed = time(0);
        // prepare filter data for conv
        for (int index = 0; index < conv_filter_count; ++index) {
            conv_filter_cpu_data[index] =
                    ((rand_r(&seed) % 200 / 200.0) - 0.5) / 2;
        }
        // prepare bias data for conv
        for (int index = 0; index < conv_bias_count; ++index) {
            conv_bias_cpu_data[index] = rand_r(&seed) % 100 / 100.0;
        }

        // prepare cpu data to converts to mlu memory
        std::vector<int16_t> conv_bias_cpu(conv_bias_count);

        // converts data format for mlu computing
        // converts conv bias data
        MGB_CNRT_CHECK(cnrtCastDataType(conv_bias_cpu_data.data(), CNRT_FLOAT32,
                                        conv_bias_cpu.data(), CNRT_FLOAT16,
                                        conv_bias_count, nullptr));

        // u should set value depending op the data or your own needs
        int filter_position = -6;
        float filter_scale = 1, filter_offset = 0;

        // count tensor nums
        int conv_input_shape[] = {ni, ci, hi, wi};
        int conv_filter_shape[] = {co, ci, kh, kw};
        int conv_bias_shape[] = {1, co, 1, 1};
        int conv_output_shape[] = {no, co, ho, wo};
        int relu_output_shape[] = {no, co, ho, wo};

        // setup tensors
        // setup conv input tensor
        conv_input_tensor = nullptr;
        MGB_CNML_CHECK(cnmlCreateTensor_V2(&conv_input_tensor, CNML_TENSOR));
        MGB_CNML_CHECK(cnmlSetTensorShape_V2(conv_input_tensor, dim_num,
                                             conv_input_shape, nullptr));
        MGB_CNML_CHECK(
                cnmlSetTensorDataType(conv_input_tensor, CNML_DATA_FLOAT16));

        // setup conv filter tensor
        cnmlTensor_t conv_filter_tensor = nullptr;
        MGB_CNML_CHECK(cnmlCreateTensor_V2(&conv_filter_tensor, CNML_FILTER));
        MGB_CNML_CHECK(cnmlSetTensorShape_V2(conv_filter_tensor, dim_num,
                                             conv_filter_shape, nullptr));
        MGB_CNML_CHECK(
                cnmlSetTensorDataType(conv_filter_tensor, CNML_DATA_FLOAT32));

        // setup conv bias tensor
        cnmlTensor_t conv_bias_tensor = nullptr;
        MGB_CNML_CHECK(cnmlCreateTensor_V2(&conv_bias_tensor, CNML_CONST));
        MGB_CNML_CHECK(cnmlSetTensorShape_V2(conv_bias_tensor, dim_num,
                                             conv_bias_shape, nullptr));
        MGB_CNML_CHECK(
                cnmlSetTensorDataType(conv_bias_tensor, CNML_DATA_FLOAT16));

        // setup conv output tensor
        cnmlTensor_t conv_output_tensor = nullptr;
        MGB_CNML_CHECK(cnmlCreateTensor_V2(&conv_output_tensor, CNML_TENSOR));
        MGB_CNML_CHECK(cnmlSetTensorShape_V2(conv_output_tensor, dim_num,
                                             conv_output_shape, nullptr));
        MGB_CNML_CHECK(
                cnmlSetTensorDataType(conv_output_tensor, CNML_DATA_FLOAT16));

        // setup relu output tensor
        relu_output_tensor = nullptr;
        MGB_CNML_CHECK(cnmlCreateTensor_V2(&relu_output_tensor, CNML_TENSOR));
        MGB_CNML_CHECK(cnmlSetTensorShape_V2(relu_output_tensor, dim_num,
                                             relu_output_shape, nullptr));
        MGB_CNML_CHECK(
                cnmlSetTensorDataType(relu_output_tensor, CNML_DATA_FLOAT16));

        // bind filters and bias to cnml const tensor
        MGB_CNML_CHECK(cnmlBindConstData_V2(
                conv_filter_tensor, conv_filter_cpu_data.data(), false));
        MGB_CNML_CHECK(cnmlBindConstData_V2(conv_bias_tensor,
                                            conv_bias_cpu.data(), false));

        // create conv param and conv op
        cnmlBaseOp_t conv_op;
        cnmlConvOpParam_t conv_param;
        // create relu op
        cnmlBaseOp_t relu_op;

        // setup conv param
        MGB_CNML_CHECK(cnmlCreateConvOpParam(&conv_param, stride_h, stride_w,
                                             dilation, dilation, pad_h, pad_w));
        // setup conv operation
        MGB_CNML_CHECK(cnmlCreateConvOp(&conv_op, conv_param, conv_input_tensor,
                                        conv_output_tensor, conv_filter_tensor,
                                        conv_bias_tensor));

        // u should set value depending op the data or your own needs
        int input_position = -6;
        float input_scale = 1, input_offset = 0;
        // prepare input tensor quant param for conv op
        cnmlQuantizedParam_t input_quant_param;
        MGB_CNML_CHECK(cnmlCreateQuantizedParam(
                &input_quant_param, input_position, input_scale, input_offset));
        // setup conv op computing datatype
        MGB_CNML_CHECK(cnmlSetOperationComputingDataType(
                conv_op, conv_input_tensor, CNML_DATA_INT8, input_quant_param));

        // prepare filter tensor quant param for conv op
        cnmlQuantizedParam_t filter_compute_quant;
        MGB_CNML_CHECK(cnmlCreateQuantizedParam(&filter_compute_quant,
                                                filter_position, filter_scale,
                                                filter_offset));
        // setup conv op computing datatype
        MGB_CNML_CHECK(cnmlSetOperationComputingDataType(
                conv_op, conv_filter_tensor, CNML_DATA_INT8,
                filter_compute_quant));

        // setup conv op computing layout
        MGB_CNML_CHECK(cnmlSetOperationComputingLayout(conv_op, CNML_NCHW));

        // setup active op using relu fuction
        MGB_CNML_CHECK(cnmlCreateActiveOp(&relu_op, CNML_ACTIVE_RELU,
                                          conv_output_tensor,
                                          relu_output_tensor));

        // setup fusion op, fuse addpad op and conv op to fusion op
        MGB_CNML_CHECK(cnmlCreateFusionOp(&fusion_op));
        MGB_CNML_CHECK(cnmlFuseOp(conv_op, fusion_op));
        MGB_CNML_CHECK(cnmlFuseOp(relu_op, fusion_op));

        MGB_CNML_CHECK(cnmlSetTensorDimMutable(conv_input_tensor,
                                               &batch_size_changable, 4));
        MGB_CNML_CHECK(cnmlSetTensorDimMutable(relu_output_tensor,
                                               &batch_size_changable, 4));

        // setup the input and output of the fusion op
        MGB_CNML_CHECK(cnmlAddFusionInput(fusion_op, conv_input_tensor));
        MGB_CNML_CHECK(cnmlAddFusionOutput(fusion_op, relu_output_tensor));

        // set operation corenum
        MGB_CNML_CHECK(cnmlSetFusionOpCorenum(fusion_op, core_num));
        // set operation coreversion
        MGB_CNML_CHECK(cnmlSetFusionOpCoreVersion(
                fusion_op, static_cast<cnmlCoreVersion_t>(core_version)));
        // set batch size changable
        MGB_CNML_CHECK(cnmlSetFusionOpBatchsizeChangable(fusion_op,
                                                         batch_size_changable));
        // compile fusion op
        MGB_CNML_CHECK(cnmlCompileFusionOp_V2(fusion_op));

        // delete tensors
        MGB_CNML_CHECK(cnmlDestroyTensor(&conv_filter_tensor));
        MGB_CNML_CHECK(cnmlDestroyTensor(&conv_bias_tensor));
        MGB_CNML_CHECK(cnmlDestroyTensor(&conv_output_tensor));

        // delete quant param
        MGB_CNML_CHECK(cnmlDestroyQuantizedParam(&input_quant_param));

        // destory filter compute quant-param
        MGB_CNML_CHECK(cnmlDestroyQuantizedParam(&filter_compute_quant));

        // delete conv params
        MGB_CNML_CHECK(cnmlDestroyConvOpParam(&conv_param));

        // delete base ops and fusion op
        MGB_CNML_CHECK(cnmlDestroyBaseOp(&conv_op));
        MGB_CNML_CHECK(cnmlDestroyBaseOp(&relu_op));
        built = true;
    }

    SmallVector<uint8_t> get_serialized_model() {
        if (!built)
            build();
        MGB_CNML_CHECK(cnmlCreateModel(&model, "mlp"));
        MGB_CNML_CHECK(cnmlAddFusionOpToModel(model, fusion_op, "subnet0"));
        std::string fname =
                ssprintf("./output/CambriconRuntimeOprTest.%s.mlu",
                         batch_size_changable ? "MutableBatchSize"
                                              : "ImmutableBatchSize");
        MGB_CNML_CHECK(cnmlSaveModel(model, fname.c_str()));
        int len = 0;
        MGB_CNRT_CHECK(cnrtGetModelSize(fname.c_str(), &len));
        SmallVector<uint8_t> buf(len);
        FILE* fstream = fopen(fname.c_str(), "rb");
        if (fstream != nullptr) {
            auto ret = fread(buf.data(), 1, len, fstream);
            mgb_assert(static_cast<int>(ret) == len);
        }
        auto fstream_close = [](FILE* fp) { fclose(fp); };
        std::unique_ptr<FILE, decltype(fstream_close)> fstream_holder{
                fstream, fstream_close};
        return std::move(buf);
    }

    void do_inference(void** input_mlu_ptrs, void** output_mlu_ptrs) {
        if (!built)
            build();
        auto&& cnrt_env = CompNodeEnv::from_comp_node(cn).cnrt_env();
        cnrt_env.activate();
        auto&& queue = cnrt_env.queue;
        cnrtNotifier_t start, end;
        MGB_CNRT_CHECK(cnrtCreateNotifier(&start));
        MGB_CNRT_CHECK(cnrtCreateNotifier(&end));
        MGB_CNRT_CHECK(cnrtPlaceNotifier(start, queue));
        MGB_CNML_CHECK(cnmlComputeFusionOpForward_V4(
                fusion_op, &conv_input_tensor, input_mlu_ptrs, 1,
                &relu_output_tensor, output_mlu_ptrs, 1, queue, nullptr));
        MGB_CNRT_CHECK(cnrtPlaceNotifier(end, queue));
        MGB_CNRT_CHECK(cnrtSyncQueue(queue));
        float time = 0.f;
        MGB_CNRT_CHECK(cnrtNotifierDuration(start, end, &time));
        printf("inference time = %.2fs\n", time * 1e-3);
        MGB_CNRT_CHECK(cnrtDestroyNotifier(&start));
        MGB_CNRT_CHECK(cnrtDestroyNotifier(&end));
    }
};
}  // namespace

TEST(TestCambriconRuntimeOpr, Basic) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    CnmlModelContext ctx{cn, false};

    // prepare parameter for addpad and conv
    const int ni = 16, ci = 64, hi = 32, wi = 32;
    const int no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<float> conv_input_cpu_data(conv_input_count);
    std::vector<float> relu_output_cpu_data(relu_output_count);

    // prepare input data for addpad
    unsigned int seed = time(0);
    for (int index = 0; index < conv_input_count; ++index) {
        conv_input_cpu_data[index] = ((rand_r(&seed) % 100 / 100.0) - 0.5) / 2;
    }

    // prepare cpu data to converts to mlu memory
    std::vector<int16_t> conv_input_cpu(conv_input_count);
    std::vector<int16_t> relu_output_cpu(relu_output_count);
    MGB_CNRT_CHECK(cnrtCastDataType(conv_input_cpu_data.data(), CNRT_FLOAT32,
                                    conv_input_cpu.data(), CNRT_FLOAT16,
                                    conv_input_count, nullptr));

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* input_mlu_ptr;
    void* output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(
            cnrtMalloc(&input_mlu_ptr, conv_input_count * sizeof(int16_t)));
    MGB_CNRT_CHECK(
            cnrtMalloc(&output_mlu_ptr, relu_output_count * sizeof(int16_t)));
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(input_mlu_ptr, conv_input_cpu.data(),
                              conv_input_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> input_holder{input_mlu_ptr,
                                                              mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> output_holder{output_mlu_ptr,
                                                               mlu_deleter};

    ctx.do_inference(&input_mlu_ptr, &output_mlu_ptr);

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(relu_output_cpu.data(), output_mlu_ptr,
                              relu_output_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtCastDataType(relu_output_cpu.data(), CNRT_FLOAT16,
                                    relu_output_cpu_data.data(), CNRT_FLOAT32,
                                    relu_output_count, nullptr));

    auto buf = ctx.get_serialized_model();
    std::shared_ptr<HostTensorND> input = std::make_shared<HostTensorND>(
            cn, TensorLayout{{ni, ci, hi, wi}, dtype::Float16()});
    memcpy(reinterpret_cast<void*>(input->ptr<dt_float16>()),
           conv_input_cpu.data(), conv_input_count * sizeof(int16_t));
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, input);
    auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                            {x}, false)[0];
    HostTensorND output(cn, {no, co, ho, wo}, dtype::Float16());
    auto func = graph->compile({make_callback_copy(y, output)});
    func->execute();
    HostTensorND out_cnml(cn, {no, co, ho, wo}, dtype::Float32()),
            out_mgb(cn, {no, co, ho, wo}, dtype::Float32());
    memcpy(out_cnml.ptr<float>(), relu_output_cpu_data.data(),
           relu_output_count * sizeof(float));
    MGB_CNRT_CHECK(cnrtCastDataType(
            reinterpret_cast<void*>(output.ptr<dt_float16>()), CNRT_FLOAT16,
            out_mgb.ptr<float>(), CNRT_FLOAT32, relu_output_count, nullptr));
    MGB_ASSERT_TENSOR_NEAR(out_cnml, out_mgb, 1e-4);
}

TEST(TestCambriconRuntimeOpr, BatchSizeChangable) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    CnmlModelContext ctx{cn, true};

    // prepare parameter for addpad and conv
    size_t ni = 16, ci = 64, hi = 32, wi = 32;
    size_t no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<float> conv_input_cpu_data(conv_input_count);
    std::vector<float> relu_output_cpu_data(relu_output_count);

    // prepare input data for addpad
    unsigned int seed = time(0);
    for (int index = 0; index < conv_input_count; ++index) {
        conv_input_cpu_data[index] = ((rand_r(&seed) % 100 / 100.0) - 0.5) / 2;
    }

    // prepare cpu data to converts to mlu memory
    std::vector<int16_t> conv_input_cpu(conv_input_count);
    std::vector<int16_t> relu_output_cpu(relu_output_count);
    MGB_CNRT_CHECK(cnrtCastDataType(conv_input_cpu_data.data(), CNRT_FLOAT32,
                                    conv_input_cpu.data(), CNRT_FLOAT16,
                                    conv_input_count, nullptr));

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* input_mlu_ptr;
    void* output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(
            cnrtMalloc(&input_mlu_ptr, conv_input_count * sizeof(int16_t)));
    MGB_CNRT_CHECK(
            cnrtMalloc(&output_mlu_ptr, relu_output_count * sizeof(int16_t)));
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(input_mlu_ptr, conv_input_cpu.data(),
                              conv_input_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> input_holder{input_mlu_ptr,
                                                              mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> output_holder{output_mlu_ptr,
                                                               mlu_deleter};

    ctx.do_inference(&input_mlu_ptr, &output_mlu_ptr);

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(relu_output_cpu.data(), output_mlu_ptr,
                              relu_output_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtCastDataType(relu_output_cpu.data(), CNRT_FLOAT16,
                                    relu_output_cpu_data.data(), CNRT_FLOAT32,
                                    relu_output_count, nullptr));
    // cnml inference finished
    {
        // change batch size
        ni = 32, no = 32;

        auto buf = ctx.get_serialized_model();
        std::shared_ptr<HostTensorND> input = std::make_shared<HostTensorND>(
                cn, TensorLayout{{ni, ci, hi, wi}, dtype::Float16()});
        memcpy(reinterpret_cast<void*>(input->ptr<dt_float16>()),
               conv_input_cpu.data(), conv_input_count * sizeof(int16_t));
        memcpy(reinterpret_cast<void*>(input->ptr<dt_float16>() +
                                       conv_input_count),
               conv_input_cpu.data(), conv_input_count * sizeof(int16_t));
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, input);
        auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(),
                                                "subnet0", {x}, true)[0];
        HostTensorND output(cn, {no, co, ho, wo}, dtype::Float16());
        auto func = graph->compile({make_callback_copy(y, output)});
        func->execute();
        HostTensorND out_cnml(cn, {no, co, ho, wo}, dtype::Float32()),
                out_mgb(cn, {no, co, ho, wo}, dtype::Float32());
        memcpy(out_cnml.ptr<float>(), relu_output_cpu_data.data(),
               relu_output_count * sizeof(float));
        memcpy(out_cnml.ptr<float>() + relu_output_count,
               relu_output_cpu_data.data(), relu_output_count * sizeof(float));
        MGB_CNRT_CHECK(cnrtCastDataType(
                reinterpret_cast<void*>(output.ptr<dt_float16>()), CNRT_FLOAT16,
                out_mgb.ptr<float>(), CNRT_FLOAT32, 2 * relu_output_count,
                nullptr));
        MGB_ASSERT_TENSOR_NEAR(out_cnml, out_mgb, 1e-4);
    }
    {
        // change batch size
        ni = 1, no = 1;
        conv_input_count = ni * hi * wi * ci;
        relu_output_count = no * ho * wo * co;

        auto buf = ctx.get_serialized_model();
        std::shared_ptr<HostTensorND> input = std::make_shared<HostTensorND>(
                cn, TensorLayout{{ni, ci, hi, wi}, dtype::Float16()});
        memcpy(reinterpret_cast<void*>(input->ptr<dt_float16>()),
               conv_input_cpu.data(), conv_input_count * sizeof(int16_t));
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, input);
        auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(),
                                                "subnet0", {x}, true)[0];
        HostTensorND output(cn, {no, co, ho, wo}, dtype::Float16());
        auto func = graph->compile({make_callback_copy(y, output)});
        func->execute();
        HostTensorND out_cnml(cn, {no, co, ho, wo}, dtype::Float32()),
                out_mgb(cn, {no, co, ho, wo}, dtype::Float32());
        memcpy(out_cnml.ptr<float>(), relu_output_cpu_data.data(),
               relu_output_count * sizeof(float));
        MGB_CNRT_CHECK(cnrtCastDataType(
                reinterpret_cast<void*>(output.ptr<dt_float16>()), CNRT_FLOAT16,
                out_mgb.ptr<float>(), CNRT_FLOAT32, relu_output_count,
                nullptr));
        MGB_ASSERT_TENSOR_NEAR(out_cnml, out_mgb, 1e-4);
    }
}

TEST(TestCambriconRuntimeOpr, Serialization) {
    using namespace serialization;
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    CnmlModelContext ctx{cn, true};
    auto buf = ctx.get_serialized_model();

    // prepare parameter for addpad and conv
    const int ni = 1, ci = 64, hi = 32, wi = 32;
    std::shared_ptr<HostTensorND> input = std::make_shared<HostTensorND>(
            cn, TensorLayout{{ni, ci, hi, wi}, dtype::Float16()});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, input);
    auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                            {x}, true)[0];
    auto fname = output_file("CambriconRuntimeOprTest");
    auto dump = [&]() {
        auto dumper = GraphDumper::make(OutputFile::make_fs(fname.c_str()));
        auto rst = dumper->dump({y});
        ASSERT_EQ(rst.outputs.size(), 1u);
    };
    auto load = [&]() {
        auto loader = GraphLoader::make(InputFile::make_fs(fname.c_str()));
        auto rst = loader->load();
        ASSERT_EQ(rst.output_var_list.size(), 1u);
    };
    dump();
    load();
}

// TODO: this test will be improved later due to peer copy for cambricon is not
// correct
TEST(TestCambriconRuntimeOpr, MultipleDevice) {
    REQUIRE_CAMBRICON_DEVICE(2);
    auto cn0 = CompNode::load("cambricon0");
    auto cn1 = CompNode::load("cambricon1");
    CnmlModelContext ctx{cn0, true};
    auto buf = ctx.get_serialized_model();

    const int ni = 8, ci = 64, hi = 32, wi = 32;

    auto graph = ComputingGraph::make();
    auto xv = std::make_shared<DeviceTensorND>(cn0, TensorShape{ni, ci, hi, wi},
                                               dtype::Float16());
    auto x = opr::SharedDeviceTensor::make(*graph, xv),
         x1 = opr::Copy::make(x, cn1);
    auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                            {x}, true)[0],
         y1 = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                             {x1}, true)[0];
    HostTensorND host_y, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y1, host_y1)});
    func->execute();
}

TEST(TestCambriconRuntimeOpr, Profiling) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    CnmlModelContext ctx{cn, true};
    auto buf = ctx.get_serialized_model();
    const int ni = 8, ci = 64, hi = 32, wi = 32;

    HostTensorGenerator<dtype::Float16, RandomDistribution::GAUSSIAN> gen(
            dt_float16(0.f), dt_float16(1.f));
    auto input = gen({ni, ci, hi, wi}, cn);
    auto graph = ComputingGraph::make();
    GraphProfiler profiler{graph.get()};
    auto x = opr::Host2DeviceCopy::make(*graph, input);
    auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                            {x}, true)[0];
    HostTensorND output;
    graph->options().var_sanity_check_first_run = false;
    auto func = graph->compile({make_callback_copy(y, output)});
    func->execute();
    profiler.to_json_full(func.get())
            ->writeto_fpath(output_file("cambricon_runtime_opr_profile.json"));
}

TEST(TestCambriconRuntimeOpr, CrossCNCopy) {
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    CnmlModelContext ctx{cn, true};

    // prepare parameter for addpad and conv
    size_t ni = 16, ci = 64, hi = 32, wi = 32;
    size_t no = 16, co = 64, ho = 32, wo = 32;

    // count tensor nums
    int conv_input_count = ni * hi * wi * ci;
    int relu_output_count = no * ho * wo * co;

    // prepare cpu origin data
    std::vector<float> conv_input_cpu_data(conv_input_count);
    std::vector<float> relu_output_cpu_data(relu_output_count);

    // prepare input data for addpad
    unsigned int seed = time(0);
    for (int index = 0; index < conv_input_count; ++index) {
        conv_input_cpu_data[index] = ((rand_r(&seed) % 100 / 100.0) - 0.5) / 2;
    }

    // prepare cpu data to converts to mlu memory
    std::vector<int16_t> conv_input_cpu(conv_input_count);
    std::vector<int16_t> relu_output_cpu(relu_output_count);
    MGB_CNRT_CHECK(cnrtCastDataType(conv_input_cpu_data.data(), CNRT_FLOAT32,
                                    conv_input_cpu.data(), CNRT_FLOAT16,
                                    conv_input_count, nullptr));

    auto mlu_deleter = [](void* p) { MGB_CNRT_CHECK(cnrtFree(p)); };
    void* input_mlu_ptr;
    void* output_mlu_ptr;

    // malloc mlu mem for fusion input and output
    MGB_CNRT_CHECK(
            cnrtMalloc(&input_mlu_ptr, conv_input_count * sizeof(int16_t)));
    MGB_CNRT_CHECK(
            cnrtMalloc(&output_mlu_ptr, relu_output_count * sizeof(int16_t)));
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(input_mlu_ptr, conv_input_cpu.data(),
                              conv_input_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::unique_ptr<void, decltype(mlu_deleter)> input_holder{input_mlu_ptr,
                                                              mlu_deleter};
    std::unique_ptr<void, decltype(mlu_deleter)> output_holder{output_mlu_ptr,
                                                               mlu_deleter};

    ctx.do_inference(&input_mlu_ptr, &output_mlu_ptr);

    // result memory copy cnml->cpu
    // memory copy cpu->mlu
    MGB_CNRT_CHECK(cnrtMemcpy(relu_output_cpu.data(), output_mlu_ptr,
                              relu_output_count * sizeof(int16_t),
                              CNRT_MEM_TRANS_DIR_DEV2HOST));
    MGB_CNRT_CHECK(cnrtCastDataType(relu_output_cpu.data(), CNRT_FLOAT16,
                                    relu_output_cpu_data.data(), CNRT_FLOAT32,
                                    relu_output_count, nullptr));
    auto cn_cpu = CompNode::load("cpu0");
    // cnml inference finished
    auto buf = ctx.get_serialized_model();
    std::shared_ptr<HostTensorND> input = std::make_shared<HostTensorND>(
            cn_cpu, TensorLayout{{ni, ci, hi, wi}, dtype::Float16()});
    memcpy(reinterpret_cast<void*>(input->ptr<dt_float16>()),
           conv_input_cpu.data(), conv_input_count * sizeof(int16_t));
    auto graph = ComputingGraph::make();
    auto host_x = opr::Host2DeviceCopy::make(*graph, input, {cn_cpu});
    auto x = opr::Copy::make(host_x, {cn});
    auto y = opr::CambriconRuntimeOpr::make(buf.data(), buf.size(), "subnet0",
                                            {x}, true)[0];
    HostTensorND output(CompNode::default_cpu(), {no, co, ho, wo},
                        dtype::Float16());
    auto func = graph->compile({make_callback_copy(y, output)});
    func->execute();
    HostTensorND out_cnml(cn_cpu, {no, co, ho, wo}, dtype::Float32()),
            out_mgb(cn_cpu, {no, co, ho, wo}, dtype::Float32());
    memcpy(out_cnml.ptr<float>(), relu_output_cpu_data.data(),
           relu_output_count * sizeof(float));
    MGB_CNRT_CHECK(
            cnrtCastDataType(reinterpret_cast<void*>(output.ptr<dt_float16>()),
                             CNRT_FLOAT16, out_mgb.ptr<float>(), CNRT_FLOAT32,
                             relu_output_count, nullptr));
    MGB_ASSERT_TENSOR_NEAR(out_cnml, out_mgb, 1e-4);
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
