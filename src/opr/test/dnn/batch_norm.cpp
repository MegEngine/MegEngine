/**
 * \file src/opr/test/dnn/batch_norm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./legacy_checker.h"
#include "megbrain/graph/bases.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/utils/timer.h"

using namespace std;
using namespace mgb;

namespace {

using Param = opr::BatchNorm::Param;

struct InputGen {

    HostTensorGenerator<> gen;
    std::unordered_map<std::string, std::pair<
        std::shared_ptr<HostTensorND>, bool>> params;

    InputGen(TensorShape tshp): gen{} {
        TensorShape pshp = {1, tshp[1], 1, 1};
        params["x"] = {gen(tshp), true};
        params["scale"] = {gen(pshp), true};
        params["bias"] = {gen(pshp), true};
        auto mean = gen(pshp), variance = gen(pshp);
        memset(mean->ptr<float>(), 0, pshp.total_nr_elems() * sizeof(float));
        memset(variance->ptr<float>(), 0, pshp.total_nr_elems() * sizeof(float));
        params["mean"] = {mean, false};
        params["variance"] = {variance, false};
    }

    std::shared_ptr<HostTensorND> get(std::string key) {
        auto iter = params.find(key);
        auto ret = gen({});
        if (iter != params.end()) {
            auto &&hv = iter->second.first;
            if (iter->second.second) {
                return hv;
            } else {
                ret->copy_from(*hv).sync();
            }
        }
        return ret;
    }
};

SymbolVarArray batch_norm_group(const SymbolVarArray& inputs, const Param &param) {
    SymbolVarArray ret;
    auto x = inputs[0], scale = inputs[1], bias = inputs[2];
         //! optional {running_mean: input[3], running_variance: inputs[4]}

    float eps = param.epsilon, avg_factor = param.avg_factor;

    auto xshp = opr::GetVarShape::make(x);
    auto tshp = opr::GetVarShape::make(scale);
    auto reduce_size = opr::reduce_prod(xshp, xshp.make_scalar(1)) /
                       opr::reduce_prod(tshp, tshp.make_scalar(1));
    auto x1 = opr::reduce_sum(x, tshp);
    auto x2 = opr::reduce_sum_sqr(x, tshp);
    auto mean = x1 / reduce_size;
    auto tmp = x2 - x1 * x1 / reduce_size;
    auto invvar = opr::PowC::make(tmp / reduce_size + eps, -0.5);
    auto ovar = (x - mean) * invvar;
    ovar = ovar * scale + bias;
    ret.push_back(ovar);

    if (inputs.size() == 3){
        ret.push_back(mean);
        ret.push_back(invvar);
    } else {
        mgb_assert(inputs.size() == 5);
        ret.push_back(opr::AddUpdate::make(inputs[3], mean,
                    {1.f - avg_factor, avg_factor}));
        ret.push_back(opr::AddUpdate::make(inputs[4], tmp / (reduce_size - 1),
                    {1.f - avg_factor, avg_factor}));
    }
    return ret;
}

SymbolVarArray batch_norm(const SymbolVarArray& inputs, const Param &param) {
    SymbolVarArray ret;
    if (inputs.size() == 3) {
        ret = opr::BatchNorm::make(inputs[0], inputs[1], inputs[2], param);
        return {ret[4], ret[2], ret[3]};
    }
    else {
        mgb_assert(inputs.size() == 5);
        ret = opr::BatchNorm::make(inputs[0], inputs[1], inputs[2],
                                   inputs[3], inputs[4], param);
        return {ret[4], ret[0], ret[1]};
    }
}

std::unique_ptr<cg::AsyncExecutable> make_func(
        const std::shared_ptr<HostTensorND> &host_x,
        const std::shared_ptr<HostTensorND> &host_scale,
        const std::shared_ptr<HostTensorND> &host_bias,
        const std::shared_ptr<HostTensorND> &host_mean,
        const std::shared_ptr<HostTensorND> &host_variance,
        const std::shared_ptr<HostTensorND> &host_y,
        const std::shared_ptr<HostTensorND> &host_grad_x,
        thin_function<SymbolVarArray(const SymbolVarArray&)> bn_func,
        bool has_statistic, bool use_fp16) {

    using Callback = thin_function<void(DeviceTensorND&)>;
    using OutputSpecItem = std::pair<SymbolVar, Callback>;
    using OutputSpec = std::vector<OutputSpecItem>;

    auto graph = ComputingGraph::make();
    auto x_raw = opr::Host2DeviceCopy::make(*graph, host_x);

    SymbolVar x;
    if (use_fp16) {
        x = opr::TypeCvt::make(x_raw, dtype::Float16(), {});
    } else {
        x = x_raw;
    }
    auto scale = opr::SharedDeviceTensor::make(*graph, *host_scale);
    auto bias = opr::SharedDeviceTensor::make(*graph, *host_bias);

    SymbolVarArray inputs{x, scale, bias};
    if (has_statistic) {
        inputs.push_back(opr::SharedDeviceTensor::make(*graph, *host_mean));
        inputs.push_back(opr::SharedDeviceTensor::make(*graph, *host_variance));
    }
    auto outputs = bn_func(inputs);
    auto y = outputs[0];
    if (use_fp16) {
        y = opr::TypeCvt::make(y, dtype::Float32(), {});
    }

    OutputSpec outspec;
    auto loss = opr::reduce_ax_sum(y.flatten(), 0);
    auto grad_x = cg::grad(loss, x_raw);
    auto scale_new = opr::AddUpdate::make(scale, cg::grad(loss, scale));
    auto bias_new = opr::AddUpdate::make(bias, cg::grad(loss, bias));

    outspec.push_back(make_callback_copy(y, *host_y));
    outspec.push_back(make_callback_copy(grad_x, *host_grad_x));
    outspec.push_back({scale_new, {}});
    outspec.push_back({bias_new, {}});
    outspec.push_back(make_callback_copy(outputs[1], *host_mean));
    outspec.push_back(make_callback_copy(outputs[2], *host_variance));

    return graph->compile(outspec);
}

TEST(TestOprDNN, BatchNormBasic)
{
    std::vector<TensorShape> input_shapes = {
        {1, 3, 10, 9},
        {2, 10, 5, 3},
        {4, 4, 12, 12}
    };

    for (auto &&has_statistic: {false, true})
    for (auto &&use_fp16: {false, true})
    for (auto &&shape : input_shapes) {
        auto input_gen = InputGen(shape);
        auto host_x = input_gen.get("x"),
             host_scale = input_gen.get("scale"),
             host_bias = input_gen.get("bias");

        auto host_mean_expected = input_gen.get("mean"),
             host_variance_expected = input_gen.get("variance"),
             host_y_expected = input_gen.get("y"),
             host_grad_x_expected = input_gen.get("grad_x");

        auto host_mean = input_gen.get("mean"),
             host_variance = input_gen.get("variance"),
             host_y = input_gen.get("y"),
             host_grad_x = input_gen.get("grad_x");

        Param param;
        param.param_dim = Param::ParamDim::DIM_1C11;
        param.avg_factor = 0.01;
        param.epsilon = 1e-4;

        using namespace std::placeholders;
        auto batch_norm_group_with_param = std::bind(batch_norm_group, _1, param);
        auto batch_norm_with_param = std::bind(batch_norm, _1, param);

        auto func_expected = make_func(host_x, host_scale, host_bias,
            host_mean_expected, host_variance_expected,
            host_y_expected, host_grad_x_expected,
            batch_norm_group_with_param, has_statistic, use_fp16);

        auto func = make_func(host_x, host_scale, host_bias,
            host_mean, host_variance, host_y, host_grad_x,
            batch_norm_with_param, has_statistic, use_fp16);

        HostTensorGenerator<> gen;
        for (size_t i = 0; i < 10; ++ i) {
            host_x->copy_from(*gen({shape})).sync();
            func_expected->execute().wait();
            func->execute().wait();
            // check running mean/var if it has statistic or check sample mean/invvar
            MGB_ASSERT_TENSOR_NEAR(*host_mean_expected, *host_mean, 1e-2);
            MGB_ASSERT_TENSOR_NEAR(*host_variance_expected, *host_variance, 1e-2);
            MGB_ASSERT_TENSOR_NEAR(*host_y_expected, *host_y, 1e-2);
            MGB_ASSERT_TENSOR_NEAR(*host_grad_x_expected, *host_grad_x, 1e-2);
        }
    }
}

}
