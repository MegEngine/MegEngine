/**
 * \file src/gopt/test/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./helper.h"
#include "megbrain/gopt/global_layout_transform.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/serialization/serializer.h"

using namespace mgb;
using namespace gopt;
using namespace serialization;

namespace {
class LayoutTransformContext : public NonCopyableObj {
public:
    using OprList = SubGraphExtractor::OprList;
    using OprFormat = Problem::OprFormat;
    using OprConfigTrait = Problem::OprConfigTrait;

    LayoutTransformContext() = delete;
    LayoutTransformContext(OprList opr_list,
                           SmallVector<TensorFormats> available_tensor_formats,
                           OprConfigTrait opr_configs)
            : m_opr_list{std::move(opr_list)},
              m_available_tensor_formats{std::move(available_tensor_formats)},
              m_opr_configs{std::move(opr_configs)} {}
    const OprList& opr_list() const { return m_opr_list; }
    const SmallVector<TensorFormats>& available_tensor_formats() const {
        return m_available_tensor_formats;
    }
    const OprConfigTrait& opr_configs() const { return m_opr_configs; }
    static std::unique_ptr<LayoutTransformContext> make() {
        OprList opr_list = {
                opr::ConvBiasForward::typeinfo(),
                opr::ConvolutionForward::typeinfo(),
                opr::ConvolutionBackwardData::typeinfo(),
                opr::ElemwiseMultiType::typeinfo(),
                opr::Elemwise::typeinfo(),
                opr::TypeCvt::typeinfo(),
                opr::PoolingForward::typeinfo(),
                opr::WarpPerspectiveForward::typeinfo(),
        };
        OprConfigTrait opr_configs;
        {
            auto& dispatchers = opr_configs[opr::ConvBias::typeinfo()];
#define cb(_fmt)                                                           \
    dispatchers[OprFormat::_fmt] =                                         \
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format( \
                    opr::ConvBias::typeinfo(), OprFormat::_fmt);
            cb(NCHW4);
            cb(NCHW32);
            cb(NHWC);
            cb(NCHW64);
            cb(CHWN4);
#undef cb
        }
        {
            auto& dispatchers =
                    opr_configs[opr::ConvolutionBackwardData::typeinfo()];
#define cb(_fmt)                                                           \
    dispatchers[OprFormat::_fmt] =                                         \
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format( \
                    opr::ConvolutionBackwardData::typeinfo(),              \
                    OprFormat::_fmt);
            cb(NCHW4);
#undef cb
        }

        {
            auto& dispatchers =
                    opr_configs[opr::ConvolutionForward::typeinfo()];
#define cb(_fmt)                                                           \
    dispatchers[OprFormat::_fmt] =                                         \
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format( \
                    opr::ConvolutionForward::typeinfo(), OprFormat::_fmt);
            cb(NCHW4);
#undef cb
        }

        {
            auto& dispatchers = opr_configs[opr::PoolingForward::typeinfo()];
#define cb(_fmt)                                                           \
    dispatchers[OprFormat::_fmt] =                                         \
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format( \
                    opr::PoolingForward::typeinfo(), OprFormat::_fmt);
            cb(NCHW4);
            cb(NCHW32);
            cb(NHWC);
            cb(NCHW64);
            cb(CHWN4);
#undef cb
        }

        {
            auto& dispatchers =
                    opr_configs[opr::WarpPerspectiveForward::typeinfo()];
#define cb(_fmt)                                                           \
    dispatchers[OprFormat::_fmt] =                                         \
            OprTensorFormatsConfiguration::find_dispatcher_by_type_format( \
                    opr::WarpPerspectiveForward::typeinfo(), OprFormat::_fmt);
            cb(NHWC);
            cb(NCHW4);
            cb(NCHW64);
#undef cb
        }

        SmallVector<TensorFormats> available_tensor_formats = {
                TensorFormats::NHWC, TensorFormats::NCHWc4,
                TensorFormats::NCHWc32, TensorFormats::NCHWc64};
        return std::make_unique<LayoutTransformContext>(
                std::move(opr_list), std::move(available_tensor_formats),
                std::move(opr_configs));
    }

private:
    OprList m_opr_list;
    SmallVector<TensorFormats> m_available_tensor_formats;
    OprConfigTrait m_opr_configs;
};
};  // namespace

#if MGB_CUDA
#if CUDA_VERSION >= 10020
TEST(TestProfiler, Conv) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);
    auto ctx = LayoutTransformContext::make();

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };
    auto x = mkvar("x", {64, 48, 14, 14},
                   dtype::Quantized4Asymm(2.5f, static_cast<uint8_t>(4)));
    auto w1 = mkcvar("w1", {48, 48, 3, 3}, dtype::QuantizedS4(2.5f));
    auto b1 = mkcvar("b1", {1, 48, 1, 1}, dtype::QuantizedS32(6.25f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::IDENTITY;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    auto c1 = opr::ConvBias::make(x, w1, b1, param, {},
                                  OperatorNodeConfig(dtype::Quantized4Asymm(
                                          12.345f, static_cast<uint8_t>(5))));
    x = opr::TypeCvt::make(c1, dtype::QuantizedS8(12.345f));
    auto w2 = mkcvar("w2", {48, 48, 3, 3}, dtype::QuantizedS8(2.5f));
    auto b2 = mkcvar("b2", {1, 48, 1, 1}, dtype::QuantizedS32(12.345f * 2.5f));
    auto c2 = opr::ConvBias::make(x, w2, b2, param, {},
                                  OperatorNodeConfig(dtype::QuantizedS8(2.5f)));

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({c2}, strategy);
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    SubGraphExtractor extractor(ctx->opr_list());
    auto partitions = extractor.extract({c2});
    ASSERT_EQ(partitions.size(), 1u);
    using Attribute = Problem::Attribute;
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW};
    Problem problem(partitions[0], ctx->available_tensor_formats(),
                    ctx->opr_configs(), attribute);
    auto profiler = ProfilerBase::make_profiler();
    auto rst = profiler->profile(problem);
    const auto& opr_rst = rst.opr_record;
    const auto& var_rst = rst.var_record;
    EXPECT_TRUE(opr_rst.count(c1.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(c2.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(x.node()->owner_opr()) > 0);
    EXPECT_TRUE(var_rst.count(w1.node()) == 0);
    EXPECT_TRUE(var_rst.count(b1.node()) == 0);
    EXPECT_TRUE(var_rst.count(w2.node()) == 0);
    EXPECT_TRUE(var_rst.count(b2.node()) == 0);
}
#endif

TEST(TestProfiler, Deconv) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);
    auto ctx = LayoutTransformContext::make();

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };
    auto x = mkvar("x", {64, 10, 7, 7}, dtype::QuantizedS8(2.5f));
    auto w1 = mkcvar("w1", {10, 10, 2, 2}, dtype::QuantizedS8(2.5f));
    using Param = opr::ConvolutionBackwardData::Param;
    Param param;
    param.format = opr::ConvolutionBackwardData::Param::Format::NCHW;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 0;
    auto c1 = opr::ConvolutionBackwardData::make(
            w1, x, param, {}, OperatorNodeConfig(dtype::QuantizedS8(2.5f)));
    auto w2 = mkcvar("w2", {10, 10, 2, 2}, dtype::QuantizedS8(2.5f));
    auto c2 = opr::ConvolutionBackwardData::make(
            w2, c1, param, {}, OperatorNodeConfig(dtype::QuantizedS8(2.5f)));

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({c2}, strategy);
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    SubGraphExtractor extractor(ctx->opr_list());
    auto partitions = extractor.extract({c2});
    ASSERT_EQ(partitions.size(), 1u);
    using Attribute = Problem::Attribute;
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW};
    Problem problem(partitions[0], ctx->available_tensor_formats(),
                    ctx->opr_configs(), attribute);
    auto profiler = ProfilerBase::make_profiler();
    auto rst = profiler->profile(problem);
    const auto& opr_rst = rst.opr_record;
    const auto& var_rst = rst.var_record;
    EXPECT_TRUE(opr_rst.count(c1.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(c2.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(x.node()->owner_opr()) > 0);
    EXPECT_TRUE(var_rst.count(w1.node()) == 0);
    EXPECT_TRUE(var_rst.count(w2.node()) == 0);
}

TEST(TestProfiler, Warp) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);
    auto ctx = LayoutTransformContext::make();

    constexpr size_t INP_H = 10, INP_W = 10, N = 16;

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };

    auto x = mkvar("x", {N, 48, INP_H, INP_W},
                   dtype::Quantized4Asymm(2.5f, static_cast<uint8_t>(4)));
    float value1 = M_PI, value2 = 0.6;
    auto gen_mat = [&](HostTensorND& mat) {
        auto ptr = mat.ptr<float>();
        for (size_t i = 0; i < N; ++i) {
            auto rot = value1, scale = value2, sheer = value1, dy = value2,
                 dx = value2, ky = value2, kx = value2, kb = value2;
            ptr[0] = ptr[4] = cos(rot) * scale;
            ptr[1] = -(ptr[3] = sin(rot) * scale);
            ptr[3] *= sheer;
            ptr[4] *= sheer;
            ptr[2] = dx;
            ptr[5] = dy;
            ptr[6] = kx;
            ptr[7] = ky;
            ptr[8] = kb;
            ptr += 9;
        }
        mgb_assert(ptr == mat.ptr<float>() + mat.shape().total_nr_elems());
    };
    auto mat_host = std::make_shared<HostTensorND>(
            x.node()->comp_node(), TensorShape{N, 3, 3}, dtype::Float32());
    gen_mat(*mat_host);
    auto mat = opr::Host2DeviceCopy::make(*graph, mat_host).rename("mat");
    TensorShape out_shp{20, 20};
    auto w1 = opr::WarpPerspectiveForward::make(x, mat, out_shp);

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({w1}, strategy);
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    SubGraphExtractor extractor(ctx->opr_list());
    auto partitions = extractor.extract({w1});
    ASSERT_EQ(partitions.size(), 1u);
    using Attribute = Problem::Attribute;
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW};
    Problem problem(partitions[0], ctx->available_tensor_formats(),
                    ctx->opr_configs(), attribute);
    auto profiler = ProfilerBase::make_profiler();
    auto rst = profiler->profile(problem);
    const auto& opr_rst = rst.opr_record;
    const auto& var_rst = rst.var_record;
    EXPECT_TRUE(opr_rst.count(w1.node()->owner_opr()) > 0);
    EXPECT_TRUE(var_rst.count(mat.node()) == 0);
    EXPECT_TRUE(var_rst.count(w1.node()->owner_opr()->input(2)) == 0);
    EXPECT_TRUE(var_rst.count(w1.node()->owner_opr()->input(0)) > 0);
}

TEST(TestProfiler, Pooling) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);
    auto ctx = LayoutTransformContext::make();

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto x = mkvar("x", {64, 64, 55, 55},
                   dtype::Quantized4Asymm(2.5f, static_cast<uint8_t>(4)));
    using Param = opr::Pooling::Param;
    Param param;
    param.format = Param::Format::NCHW;
    auto p1 = opr::Pooling::make(x, param);
    x = opr::TypeCvt::make(p1, dtype::QuantizedS8(12.345f));
    auto p2 = opr::Pooling::make(x, param);

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({p2}, strategy);
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    SubGraphExtractor extractor(ctx->opr_list());
    auto partitions = extractor.extract({p2});
    ASSERT_EQ(partitions.size(), 1u);
    using Attribute = Problem::Attribute;
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW};
    Problem problem(partitions[0], ctx->available_tensor_formats(),
                    ctx->opr_configs(), attribute);
    auto profiler = ProfilerBase::make_profiler();
    auto rst = profiler->profile(problem);
    const auto& opr_rst = rst.opr_record;
    EXPECT_TRUE(opr_rst.count(p1.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(p2.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(x.node()->owner_opr()) > 0);
}

TEST(TestProfiler, Elemwise) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);
    auto ctx = LayoutTransformContext::make();

    HostTensorGenerator<dtype::Int8> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto a = mkvar("a", {64, 48, 14, 14}, dtype::Float32());
    auto b = mkvar("b", {1, 48, 1, 1}, dtype::Float32());
    auto c = opr::Elemwise::make({a, b},
                                 {opr::Elemwise::Param::Mode::FUSE_ADD_RELU});
    auto q4c = opr::TypeCvt::make(
            c, dtype::Quantized4Asymm(2.5f, static_cast<uint8_t>(4)));
    auto q8a = mkvar("q8a", {64, 48, 14, 14}, dtype::QuantizedS8(2.5f));
    auto q8b = mkvar("q8b", {64, 48, 14, 14}, dtype::QuantizedS8(1.2f));
    auto q8d = opr::ElemwiseMultiType::make(
            {q8a, q8b}, {opr::ElemwiseMultiType::Param::Mode::QFUSE_ADD_RELU},
            OperatorNodeConfig(dtype::QuantizedS8(12.f)));
    auto q4d = opr::TypeCvt::make(
            q8d, dtype::Quantized4Asymm(1.2f, static_cast<uint8_t>(3)));
    auto q4e = opr::ElemwiseMultiType::make(
            {q4c, q4d}, {opr::ElemwiseMultiType::Param::Mode::QADD},
            OperatorNodeConfig(
                    dtype::Quantized4Asymm(13.f, static_cast<uint8_t>(4))));

    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    SubGraphExtractor extractor(ctx->opr_list());
    auto partitions = extractor.extract({q4e});
    ASSERT_EQ(partitions.size(), 1u);
    using Attribute = Problem::Attribute;
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW};
    Problem problem(partitions[0], ctx->available_tensor_formats(),
                    ctx->opr_configs(), attribute);
    auto profiler = ProfilerBase::make_profiler();
    auto rst = profiler->profile(problem);
    const auto& opr_rst = rst.opr_record;
    const auto& var_rst = rst.var_record;
    EXPECT_TRUE(opr_rst.count(c.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(q8d.node()->owner_opr()) > 0);
    EXPECT_TRUE(opr_rst.count(q4e.node()->owner_opr()) > 0);
    EXPECT_TRUE(var_rst.count(a.node()) > 0);
    EXPECT_TRUE(var_rst.count(b.node()) > 0);
    EXPECT_TRUE(var_rst.count(q8a.node()) > 0);
    EXPECT_TRUE(var_rst.count(q8b.node()) > 0);
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
