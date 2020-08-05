/**
 * \file src/tensorrt/test/make_trt_net.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/opr/basic_arith.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/debug.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"
#include "make_trt_net.h"

#include <random>

using namespace mgb;
using namespace opr;
using namespace nvinfer1;


intl::SimpleTensorRTNetwork::SimpleTensorRTNetwork() {
    host_x = gen({5, 23, 28, 28});
    host_w = gen({32, 23, 3, 3});
    host_b = gen({1, 32, 1, 1});

    graph = ComputingGraph::make();
    x = Host2DeviceCopy::make(*graph, host_x);
    auto w = Host2DeviceCopy::make(*graph, host_w),
         b = Host2DeviceCopy::make(*graph, host_b),
         y0 = opr::Convolution::make(x, w);
    y = y0 + b;
}

std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
intl::SimpleTensorRTNetwork::create_trt_network(bool has_batch_dim) {
    CompNode::load("xpu0").activate();
    Weights wt_filter{DataType::kFLOAT, nullptr, 0},
            wt_bias{DataType::kFLOAT, nullptr, 0};
    wt_filter.type = DataType::kFLOAT;
    wt_bias.type = DataType::kFLOAT;
    wt_filter.values = host_w->raw_ptr();
    wt_bias.values = host_b->raw_ptr();
    wt_filter.count = host_w->shape().total_nr_elems();
    wt_bias.count = host_b->shape().total_nr_elems();
    auto builder = createInferBuilder(TensorRTOpr::Logger::instance());
#if NV_TENSOR_RT_VERSION >= 6001
    nvinfer1::NetworkDefinitionCreationFlags flags;
    ::memset(&flags, 0, sizeof(nvinfer1::NetworkDefinitionCreationFlags));
    if (has_batch_dim)
        flags = 1 << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::
                                              kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flags);
#else
    auto network = builder->createNetwork();
#endif
    nvinfer1::ITensor* data;
#if NV_TENSOR_RT_VERSION >= 6001
    if (has_batch_dim) {
        data = network->addInput("data", DataType::kFLOAT,
                                 Dims4{5, 23, 28, 28});
    } else {
        data = network->addInput("data", DataType::kFLOAT, Dims3{23, 28, 28});
    }
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        data->setAllowedFormats(formats);
    }
#else
    if (has_batch_dim) {
        data = network->addInput("data", DataType::kFLOAT,
                                 DimsNCHW{5, 23, 28, 28});
    } else {
        data = network->addInput("data", DataType::kFLOAT, DimsCHW{23, 28, 28});
    }
#endif
    mgb_assert(data != nullptr, "data is invalid");
    auto conv1 = network->addConvolution(*data, 32, DimsHW{3, 3}, wt_filter,
                                         wt_bias);
    mgb_assert(conv1 != nullptr, "conv1 is invalid");
    conv1->setStride(DimsHW{1, 1});
    conv1->getOutput(0)->setName("prob");
    network->markOutput(*conv1->getOutput(0));
#if NV_TENSOR_RT_VERSION >= 6001
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        conv1->getOutput(0)->setAllowedFormats(formats);
    }
#endif

    return std::make_pair(builder, network);
}

intl::SimpleQuantizedTensorRTNetwork::SimpleQuantizedTensorRTNetwork() {
    host_x = range_gen({32, 8, 28, 28});
    host_w = weight_gen({8, 8, 3, 3});
    host_b = range_gen({1, 8, 1, 1});

    {
        float* ptr = reinterpret_cast<float*>(host_w->raw_ptr());
        ptr[0] = -127*1.1f;
        ptr[1] = 127*1.1f;
    }

    graph = ComputingGraph::make();
    auto mkvar = [this](const char* name,
                        const std::shared_ptr<HostTensorND>& host_ts,
                        const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, host_ts).rename(name),
                dtype);
    };
    auto mkcvar = [this](const char* name,
                         const std::shared_ptr<HostTensorND>& host_ts,
                         const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *host_ts).rename(name),
                dtype);
    };

    x = mkvar("x", host_x, dtype::Float32());
    quantized_x = mkvar("quantized_x", host_x, dtype::QuantizedS8(1.2f));
    auto float_w = mkcvar("float_w", host_w, dtype::Float32()),
         float_b = mkcvar("float_b", host_b, dtype::Float32()),
         w = opr::TypeCvt::make(float_w, dtype::QuantizedS8(1.1f)),
         b = opr::TypeCvt::make(float_b, dtype::QuantizedS32(1.2f * 1.1f));

    {
        auto xshp = opr::GetVarShape::make(quantized_x);

        auto cv = [this](int v) { return quantized_x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        quantized_x = opr::Reshape::make(quantized_x, tshp);
        quantized_x = opr::Dimshuffle::make(quantized_x, {0, 1, 3, 4, 2});
    }

    {
        auto wshp = opr::GetVarShape::make(w);

        auto cv = [&w](int v) { return w.make_scalar(v); };
        auto sub = [&wshp, &cv](int idx) {
            return opr::IndexAt::make(wshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        w = opr::Reshape::make(w, tshp);
        w = opr::Dimshuffle::make(w, {0, 1, 3, 4, 2});
    }

    {
        auto bshp = opr::GetVarShape::make(b);

        auto cv = [&b](int v) { return b.make_scalar(v); };
        auto sub = [&bshp, &cv](int idx) {
            return opr::IndexAt::make(bshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        b = opr::Reshape::make(b, tshp);
        b = opr::Dimshuffle::make(b, {0, 1, 3, 4, 2});
    }

    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::IDENTITY;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;

    quantized_y =
            opr::ConvBias::make(quantized_x, w, b, param, {},
                                OperatorNodeConfig{dtype::QuantizedS8(1.1f)});
    param.format = opr::ConvBias::Param::Format::NCHW;
    y = opr::ConvBias::make(x, float_w, float_b, param, {},
                            OperatorNodeConfig{dtype::Float32()});

    auto yshp = opr::GetVarShape::make(quantized_y);

    auto cv = [this](int v) { return quantized_y.make_scalar(v); };
    auto sub = [&yshp, &cv](int idx) {
        return opr::IndexAt::make(yshp, {{0, cv(idx)}});
    };
    auto tshp = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
    quantized_y = opr::Dimshuffle::make(quantized_y, {0, 1, 4, 2, 3});
    quantized_y = opr::Reshape::make(quantized_y, tshp);
    quantized_y = TypeCvt::make(quantized_y, dtype::Float32());
}

std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
intl::SimpleQuantizedTensorRTNetwork::create_trt_network(
        bool has_batch_dim) {
    CompNode::load("xpu0").activate();
    Weights wt_filter{DataType::kFLOAT, nullptr, 0},
            wt_bias{DataType::kFLOAT, nullptr, 0};
    wt_filter.type = DataType::kFLOAT;
    wt_bias.type = DataType::kFLOAT;
    wt_filter.values = host_w->raw_ptr();
    wt_bias.values = host_b->raw_ptr();
    wt_filter.count = host_w->shape().total_nr_elems();
    wt_bias.count = host_b->shape().total_nr_elems();
    auto builder = createInferBuilder(TensorRTOpr::Logger::instance());
#if NV_TENSOR_RT_VERSION >= 6001
    nvinfer1::NetworkDefinitionCreationFlags flags;
    ::memset(&flags, 0, sizeof(nvinfer1::NetworkDefinitionCreationFlags));
    if (has_batch_dim)
        flags = 1 << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::
                                              kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flags);
#else
    auto network = builder->createNetwork();
#endif
    nvinfer1::ITensor* data;
#if NV_TENSOR_RT_VERSION >= 6001
    if (has_batch_dim) {
        data = network->addInput("data", DataType::kFLOAT,
                                 Dims4{32, 8, 28, 28});
    } else {
        data = network->addInput("data", DataType::kFLOAT, Dims3{8, 28, 28});
    }
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        data->setAllowedFormats(formats);
    }
#else
    if (has_batch_dim) {
        data = network->addInput("data", DataType::kFLOAT,
                                 DimsNCHW{32, 8, 28, 28});
    } else {
        data = network->addInput("data", DataType::kFLOAT, DimsCHW{8, 28, 28});
    }
#endif
    data->setDynamicRange(-127.f * 1.2f, 127.f * 1.2f);
    mgb_assert(data != nullptr, "data is invalid");
    auto add_conv = [&](const char* name, nvinfer1::ITensor* inp) {
        auto conv = network->addConvolution(*inp, 8, DimsHW{3, 3}, wt_filter,
                                            wt_bias);
        mgb_assert(conv != nullptr, "conv1 is invalid");
        conv->setName(name);
        conv->setStride(DimsHW{1, 1});
        conv->setPadding(DimsHW{1, 1});
        conv->getOutput(0)->setDynamicRange(-127.f * 1.1f, 127.f * 1.1f);
        // conv->setPrecision(nvinfer1::DataType::kINT8);
        return conv->getOutput(0);
    };
    auto out = add_conv("conv1", data);
    out->setName("prob");
#if NV_TENSOR_RT_VERSION >= 6001
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        out->setAllowedFormats(formats);
    }
#endif
    network->markOutput(*out);

    return std::make_pair(builder, network);
}

intl::ConcatConvTensorRTNetwork::ConcatConvTensorRTNetwork() {
    host_x0 = gen({5, 23, 14, 28});
    host_x1 = gen({5, 23, 14, 28});
    host_w = gen({32, 46, 3, 3});
    host_b = gen({1, 32, 1, 1});

    graph = ComputingGraph::make();
    x0 = Host2DeviceCopy::make(*graph, host_x0);
    x1 = Host2DeviceCopy::make(*graph, host_x1);
    auto y0 = opr::Concat::make({x0, x1}, 1),
         w = Host2DeviceCopy::make(*graph, host_w),
         b = Host2DeviceCopy::make(*graph, host_b),
         y1 = opr::Convolution::make(y0, w);
    y = y1 + b;
}

std::pair<nvinfer1::IBuilder*, INetworkDefinition*>
intl::ConcatConvTensorRTNetwork::create_trt_network(bool has_batch_dim) {
    CompNode::load("xpu0").activate();
    auto builder = createInferBuilder(TensorRTOpr::Logger::instance());
#if NV_TENSOR_RT_VERSION >= 6001
    nvinfer1::NetworkDefinitionCreationFlags flags;
    ::memset(&flags, 0, sizeof(nvinfer1::NetworkDefinitionCreationFlags));
    if (has_batch_dim) flags = 1 << static_cast<int>(
                    nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(flags);
#else
    auto network = builder->createNetwork();
#endif
    ITensor *data0, *data1;
#if NV_TENSOR_RT_VERSION >= 6001
    if (has_batch_dim) {
        data0 = network->addInput("x0", DataType::kFLOAT,
                                  Dims4{5, 23, 14, 28});
        data1 = network->addInput("x1", DataType::kFLOAT,
                                  Dims4{5, 23, 14, 28});
    } else {
        data0 = network->addInput("x0", DataType::kFLOAT, Dims3{23, 14, 28});
        data1 = network->addInput("x1", DataType::kFLOAT, Dims3{23, 14, 28});
    }
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        data0->setAllowedFormats(formats);
        data1->setAllowedFormats(formats);
    }
#else
    if (has_batch_dim) {
        data0 = network->addInput("x0", DataType::kFLOAT,
                                  DimsNCHW{5, 23, 14, 28});
        data1 = network->addInput("x1", DataType::kFLOAT,
                                  DimsNCHW{5, 23, 14, 28});
    } else {
        data0 = network->addInput("x0", DataType::kFLOAT, DimsCHW{23, 14, 28});
        data1 = network->addInput("x1", DataType::kFLOAT, DimsCHW{23, 14, 28});
    }
#endif
    ITensor* inputTensors[] = {data0, data1};
    auto concat = network->addConcatenation(inputTensors, 2);
    mgb_assert(concat != nullptr, "concat is null!");
    concat->setName("concat0");
    if (has_batch_dim) {
        concat->setAxis(1);
    } else {
        concat->setAxis(0);
    }

    Weights wt_filter{DataType::kFLOAT, host_w->raw_ptr(), 0},
            wt_bias{DataType::kFLOAT, host_b->raw_ptr(), 0};
    wt_filter.count = host_w->shape().total_nr_elems();
    wt_bias.count = host_b->shape().total_nr_elems();
    auto conv1 = network->addConvolution(*concat->getOutput(0), 32,
                                         DimsHW{3, 3}, wt_filter, wt_bias);
    mgb_assert(conv1 != nullptr, "conv1 is invalid");
    conv1->setName("conv1");
    conv1->setStride(DimsHW{1, 1});
    conv1->getOutput(0)->setName("convOut");
    network->markOutput(*conv1->getOutput(0));
#if NV_TENSOR_RT_VERSION >= 6001
    {
        nvinfer1::TensorFormats formats =
                1 << static_cast<int>(nvinfer1::TensorFormat::kLINEAR);
        conv1->getOutput(0)->setAllowedFormats(formats);
    }
#endif
    return std::make_pair(builder, network);
}

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
