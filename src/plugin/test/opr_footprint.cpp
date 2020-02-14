/**
 * \file src/plugin/test/opr_footprint.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/json.h"

using namespace mgb;

namespace {
json::Object& visit_json_obj(json::Object& obj, const std::string& key) {
    auto&& val = obj[key];
    mgb_assert(val, "key %s not found", key.c_str());
    return static_cast<json::Object&>(*val);
};
void compile_and_run(std::shared_ptr<ComputingGraph> graph, SymbolVar& out,
                     HostTensorND& host_out, uint64_t computation,
                     uint64_t memory) {
    graph->options().graph_opt_level = 0;
    auto func = graph->compile({make_callback_copy(out, host_out)});
    auto profiler = std::make_shared<GraphProfiler>(graph.get());
    func->execute();
    host_out.sync();

    auto&& opr = out.node()->owner_opr();
    auto root_ptr = profiler->to_json();
    auto&& json_rst = *root_ptr;
    auto&& opr_fp_rst = visit_json_obj(json_rst, "opr_footprint");
    auto&& opr_fp_item = visit_json_obj(opr_fp_rst, opr->id_str());

    uint64_t mem_rst =
            static_cast<json::NumberInt*>(opr_fp_item["memory"].get())
                    ->get_impl();
    uint64_t comp_rst =
            static_cast<json::NumberInt*>(opr_fp_item["computation"].get())
                    ->get_impl();

    ASSERT_EQ(memory, mem_rst);
    ASSERT_EQ(computation, comp_rst);
}

template <typename Func, typename DType, typename Param>
void run_test(Func func, std::initializer_list<size_t>&& host_x_shape,
              std::initializer_list<size_t>&& host_y_shape,
              std::initializer_list<size_t>&& host_z_shape,
              uint64_t computation, uint64_t nr_elems, DType dtype,
              const Param& param, CompNode cn = CompNode::load("xpux")) {
    HostTensorGenerator<DType> gen;
    auto host_x = gen(host_x_shape, cn);
    auto host_y = gen(host_y_shape, cn);
    auto host_z = gen(host_z_shape, cn);

    auto graph = ComputingGraph::make();
    SymbolVar x = opr::SharedDeviceTensor::make(*graph, *host_x.get())
                          .rename("x"),
              y = opr::SharedDeviceTensor::make(*graph, *host_y.get())
                          .rename("y"),
              z = opr::SharedDeviceTensor::make(*graph, *host_z.get())
                          .rename("z"),
              f = func(x, y, z, param);

    HostTensorND host_f;
    compile_and_run(graph, f, host_f, computation, dtype.size(nr_elems));
}

template <class Param, typename Func>
void test_conv_group(size_t n, size_t ic, size_t oc, size_t ih, size_t iw,
                     size_t fh, size_t fw, size_t ph, size_t pw, size_t sh,
                     size_t sw, Func func) {
    Param param;

    size_t ow = (iw + 2 * pw - fw) / sw + 1;
    size_t oh = (ih + 2 * ph - fh) / sh + 1;

    uint64_t computation = n * ic * oc * ow * oh * fw * fh * 2;
    uint64_t memory = n * ic * ih * iw + oc * ic * fw * fh + n * oc * oh * ow;

    param.stride_h = sh;
    param.stride_w = sw;
    param.pad_h = ph;
    param.pad_w = pw;

    run_test(func, {n, ic, ih, iw}, {oc, ic, fh, fw}, {n, oc, oh, ow},
             computation, memory, dtype::Float32(), param);
};

template <class Param, typename Func>
void test_conv_bias_group_nchw4(size_t n, size_t ic, size_t oc, size_t ih,
                                size_t iw, size_t fh, size_t fw, size_t ph,
                                size_t pw, size_t sh, size_t sw, Func func,
                                size_t group) {
    Param param;

    size_t ow = (iw + 2 * pw - fw) / sw + 1;
    size_t oh = (ih + 2 * ph - fh) / sh + 1;

    uint64_t computation =
            (n * ic * oc * ow * oh * fw * fh * 2 + n * oc * ow * oh) * group;
    uint64_t memory =
            (n * ic * ih * iw + oc * ic * fw * fh + n * oc * oh * ow + 4 * oc) *
            group;

    param.stride_h = sh;
    param.stride_w = sw;
    param.pad_h = ph;
    param.pad_w = pw;
    param.format = Param::Format::NCHW4;

    if (group == 1) {
        run_test(func, {n, group * ic / 4, ih, iw, 4}, {oc, ic / 4, fh, fw, 4},
                 {1, oc * group / 4, 1, 1, 4}, computation, memory,
                 dtype::QuantizedS8(1.0f), param, CompNode::load("cpux"));
    } else {
        param.sparse = Param::Sparse::GROUP;
        run_test(func, {n, group * ic / 4, ih, iw, 4},
                 {group, oc, ic / 4, fh, fw, 4}, {1, oc * group / 4, 1, 1, 4},
                 computation, memory, dtype::QuantizedS8(1.0f), param,
                 CompNode::load("cpux"));
    }
}

}  // namespace

TEST(TestOprFootprint, Elemwise) {
    using Param = opr::Elemwise::Param;
    auto test_elemwise_group = [](Param::Mode mode, size_t nr_inputs,
                                  size_t k) {
        auto func = [&nr_inputs](SymbolVar x, SymbolVar y, SymbolVar z,
                                 const Param& param = {}) {
            SymbolVarArray inputs{x, y, z};
            inputs.resize(nr_inputs);
            return opr::Elemwise::make(inputs, param);
        };
        Param param;
        param.mode = mode;
        run_test(func, {2, 3, 3}, {2, 3, 3}, {2, 3, 3}, 18 * k,
                 18 * (nr_inputs + 1), dtype::Float32(), param);
        auto mem = 30 * (nr_inputs + 1);
        if (nr_inputs == 3)
            mem -= 2 * 3 * 4;
        run_test(func, {2, 5, 3}, {2, 5, 3}, {2, 1, 3}, 30 * k, mem,
                 dtype::Int32(), param);
    };
    test_elemwise_group(Param::Mode::SIGMOID, 1, 1);
    test_elemwise_group(Param::Mode::ADD, 2, 1);
    test_elemwise_group(Param::Mode::FUSE_MUL_ADD3, 3, 2);
}

TEST(TestOprFootprint, AddUpdate) {
    using Param = opr::AddUpdate::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z,
                   const Param& param = {}) {
        return opr::AddUpdate::make(x, y, param);
    };
    Param param;
    run_test(func, {2, 3, 3}, {2, 3, 3}, {0}, 18 * 3, 18 * 3, dtype::Float32(),
             param);
    run_test(func, {2, 3, 5}, {2, 3, 5}, {0}, 30 * 3, 30 * 3, dtype::Int16(),
             param);
}

TEST(TestOprFootprint, ConvolutionForward) {
    using OprType = opr::ConvolutionForward;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make(x, y, param);
    };
    REQUIRE_GPU(1);
    test_conv_group<Param, decltype(func)>
            //    n, ic, oc, ih, iw, fh, fw, ph, pw, sh, sw
            (10, 3, 2, 24, 24, 3, 3, 1, 1, 3, 3, func);
    test_conv_group<Param, decltype(func)>(20, 4, 3, 48, 24, 3, 5, 2, 2, 2, 2,
                                           func);
}

TEST(TestOprFootprint, ConvolutionBackwardData) {
    using OprType = opr::ConvolutionBackwardData;
    using Param = OprType::Param;
    auto func = [](SymbolVar src_for_shp, SymbolVar filter, SymbolVar diff,
                   const Param& param) {
        return OprType::make(filter, diff, src_for_shp, param);
    };
    //    n, ic, oc, ih, iw, fh, fw, ph, pw, sh, sw
    test_conv_group<opr::ConvolutionForward::Param, decltype(func)>(
            10, 3, 2, 24, 24, 3, 3, 1, 1, 3, 3, func);
    test_conv_group<opr::ConvolutionForward::Param, decltype(func)>(
            20, 4, 3, 48, 24, 3, 5, 2, 2, 2, 2, func);
}

TEST(TestOprFootprint, ConvolutionBackwardFilter) {
    using OprType = opr::ConvolutionBackwardFilter;
    using Param = OprType::Param;
    auto func = [](SymbolVar src, SymbolVar filter, SymbolVar diff,
                   const Param& param) {
        return OprType::make(src, diff, filter, param);
    };
    //    n, ic, oc, ih, iw, fh, fw, ph, pw, sh, sw
    test_conv_group<Param, decltype(func)>(10, 3, 2, 24, 24, 3, 3, 1, 1, 3, 3,
                                           func);
    test_conv_group<Param, decltype(func)>(20, 4, 3, 48, 24, 3, 5, 2, 2, 2, 2,
                                           func);
}

TEST(TestOprFootprint, MatrixMul) {
    using OprType = opr::MatrixMul;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make(x, y, param);
    };
    run_test(func, {3, 5}, {5, 7}, {0}, 3 * 5 * 7 * 2, 3 * 5 + 5 * 7 + 3 * 7,
             dtype::Float32(), Param{});
    run_test(func, {7, 3}, {8, 7}, {0}, 3 * 7 * 8 * 2, 3 * 7 + 8 * 7 + 3 * 8,
             dtype::Float32(), Param{true, true});
}

TEST(TestOprFootprint, PoolingForward) {
    using OprType = opr::PoolingForward;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make(x, param);
    };
    Param param;
    param.window_h = param.stride_h = 2;
    param.window_w = param.stride_w = 3;
    run_test(func, {10, 7, 8, 6}, {0}, {0}, 10 * 7 * 8 * 6,
             10 * 7 * (8 * 6 + 4 * 3), dtype::Float32(), Param{});
}

TEST(TestOprFootprint, Concat) {
    using OprType = opr::Concat;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make({x, y, z}, param.axis);
    };
    Param param;
    run_test(func, {1, 3, 5}, {2, 3, 5}, {3, 3, 5}, 6 * 3 * 5, 6 * 3 * 5 * 2,
             dtype::Float32(), param);
}

TEST(TestOprFootprint, Reduce) {
    using OprType = opr::Reduce;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make(x, param);
    };
    Param param;
    param.axis = 1;
    run_test(func, {5, 3, 3}, {0}, {0}, 5 * 3 * 3, 5 * 3 * 3 + 5 * 3,
             dtype::Float32(), param);
}

TEST(TestOprFootprint, Dimshuffle) {
    using OprType = opr::Dimshuffle;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        return OprType::make(x, {1, 2, 0}, 0);
    };
    run_test(func, {2, 3, 5}, {3, 5, 2}, {0}, 2 * 3 * 5, 2 * 3 * 5 * 2,
             dtype::Float32(), Param());
}

TEST(TestOprFootprint, Host2DeviceCopy) {
    using OprType = opr::Host2DeviceCopy;
    REQUIRE_GPU(1);
    auto&& cpu = CompNode::load("cpu1");
    auto float32 = dtype::Float32();
    auto data = std::make_shared<HostTensorND>(
            HostTensorND(cpu, {2, 3, 5}, float32));
    auto graph = ComputingGraph::make();
    auto out_var = OprType::make_no_value_infer(*graph.get(), data);
    HostTensorND host_out(cpu, float32);
    compile_and_run(graph, out_var, host_out, 2 * 3 * 5,
                    float32.size(2 * 3 * 5));
}

TEST(TestOprFootprint, NCHW4Convolution) {
    using OprType = opr::ConvBias;
    using Param = OprType::Param;
    auto func = [](SymbolVar x, SymbolVar y, SymbolVar z, const Param& param) {
        x = opr::TypeCvt::make(x, dtype::QuantizedS8(1.3f));
        y = opr::TypeCvt::make(y, dtype::QuantizedS8(1.4f));
        z = opr::TypeCvt::make(z, dtype::QuantizedS32(1.3f * 1.4f));
        return OprType::make(x, y, z, param, {},
                             OperatorNodeConfig{dtype::QuantizedS8(0.6f)});
    };
    test_conv_bias_group_nchw4<Param, decltype(func)>(10, 4, 8, 24, 24, 3, 3, 1,
                                                      1, 3, 3, func, 1);
    test_conv_bias_group_nchw4<Param, decltype(func)>(20, 4, 4, 48, 24, 3, 5, 2,
                                                      3, 2, 1, func, 4);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
