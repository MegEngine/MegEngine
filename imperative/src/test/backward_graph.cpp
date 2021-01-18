/**
 * \file imperative/src/test/backward_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/backward_graph_opt.h"

using namespace mgb;
using namespace cg;
using namespace imperative;

template <typename T>
T prepare_backward_graph_inputs(const BackwardGraphResult& bg, const T& inputs, const T& outputs, const T& grads) {
    T ret;
    size_t i = 0;
    for (auto&& t : inputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : outputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : grads) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    return ret;
}

template <typename T, typename U>
T expand_grads(const U& bg, const T& outputs) {
    T ret(bg.input_has_grad.size());
    for (size_t i = 0, j = 0; i < bg.input_has_grad.size(); ++i) {
        if (bg.input_has_grad[i]) {
            ret[i] = outputs[j++];
        }
    }
    return ret;
}

template <typename T>
T prepare_optimized_backward_inputs(const OptimizedBackwardGraphResult& bg, const T& precomp, const T& inputs, const T& outputs, const T& grads) {
    T ret = precomp;
    size_t i = 0;
    for (auto&& t : inputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : outputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : grads) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    return ret;
}

TEST(TestImperative, BackwardGraphBasic) {
    HostTensorGenerator<> gen;
    SmallVector<HostTensorND> hvs;
    SmallVector<TensorPtr> inputs;
    for(size_t i = 0; i < 2; ++ i) {
        hvs.push_back(*gen({42}));
        inputs.push_back(Tensor::make(hvs.back()));
    }

    using Param = opr::Elemwise::Param;
    Param param{Param::Mode::MUL};
    auto attr = OprAttr::make("Elemwise");
    attr->cast_final_safe<OprAttr>().param.write_pod(param);

    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& i : inputs) {
        input_descs.push_back({i->layout(), i->comp_node()});
    }
    auto result = OpDef::make_backward_graph(*attr, input_descs, {true, true}, {true});
    auto&& save_for_backward = result.save_for_backward;
    auto&& input_has_grad = result.input_has_grad;

    auto outputs = OpDef::apply_on_physical_tensor(*attr, inputs);
    inputs.push_back(outputs[0]);
    hvs.push_back(*gen({42}));
    inputs.push_back(Tensor::make(hvs.back()));
    mgb_assert(save_for_backward.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        if (!save_for_backward[i]) {
            inputs[i].reset(); // drop unused tensor
        }
    }
    SmallVector<TensorPtr> backward_graph_inputs;
    for (auto&& i : inputs) {
        if (i) {
            backward_graph_inputs.push_back(i);
        }
    }
    inputs.clear();
    auto input_grads = OpDef::apply_on_physical_tensor(*(result.backward), backward_graph_inputs);
    mgb_assert(input_grads.size() == input_has_grad.size());
    for (size_t i = 0; i < input_has_grad.size(); ++ i) {
        mgb_assert(input_has_grad[i] == static_cast<bool>(input_grads[i]));
    }

    SmallVector<HostTensorND> res;
    for (auto&& i : input_grads) {
        res.emplace_back();
        res.back().copy_from(i->dev_tensor()).sync();
    }
    for (size_t i = 0; i < 42; ++ i) {
        for (size_t j = 0; j < 1; ++ j) {
            ASSERT_EQ(hvs[2].ptr<float>()[i] * hvs[j].ptr<float>()[i], res[j ^ 1].ptr<float>()[i]);
        }
    }
}

TEST(TestImperative, BackwardGraphIdentity) {
    HostTensorGenerator<> gen;
    auto host_a = gen({42}), host_dc = gen({42});
    auto a = Tensor::make(*host_a), dc = Tensor::make(*host_dc);
    SmallVector<TensorPtr> inputs;
    inputs.push_back(a);

    auto attr = OprAttr::make("Identity");
    attr->cast_final_safe<OprAttr>().param.write_pod<megdnn::param::Empty>({});

    SmallVector<LogicalTensorDesc> input_descs;
    input_descs.push_back({a->layout(), a->comp_node()});
    auto result = OpDef::make_backward_graph(*attr, input_descs, {true}, {true});
    auto&& save_for_backward = result.save_for_backward;
    auto&& input_has_grad = result.input_has_grad;

    auto outputs = OpDef::apply_on_physical_tensor(*attr, inputs);
    inputs.push_back(outputs[0]);
    inputs.push_back(dc);
    mgb_assert(save_for_backward.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i) {
        if (!save_for_backward[i]) {
            inputs[i].reset(); // drop unused tensor
        }
    }
    SmallVector<TensorPtr> backward_graph_inputs;
    for (auto&& i : inputs) {
        if (i) {
            backward_graph_inputs.push_back(i);
        }
    }
    inputs.clear();
    auto input_grads = OpDef::apply_on_physical_tensor(*(result.backward), backward_graph_inputs);
    mgb_assert(input_grads.size() == input_has_grad.size());
    for (size_t i = 0; i < input_has_grad.size(); ++ i) {
        mgb_assert(input_has_grad[i] == static_cast<bool>(input_grads[i]));
    }

    HostTensorND hv;
    hv.copy_from(input_grads[0]->dev_tensor()).sync();
    for (size_t i = 0; i < 42; ++ i) {
        ASSERT_EQ(host_dc->ptr<float>()[i], hv.ptr<float>()[i]);
    }
}

TEST(TestImperative, BatchNormGrad) {
    auto cn = CompNode::load("xpux");
    using Param = opr::BatchNorm::Param;
    size_t N=2, C=3, H=5, W=5;
    LogicalTensorDesc inp{TensorLayout{{N, C, H, W}, dtype::Float32()}, cn};
    LogicalTensorDesc stat{TensorLayout{{C}, dtype::Float32()}, cn};
    {
        auto op = OprAttr::make("BatchNorm");
        auto&& attr = op->cast_final_safe<OprAttr>();
        Param param;
        param.fwd_mode = Param::FwdMode::TRAINING;
        attr.param.write_pod(param);
        OpDef::make_backward_graph(attr, {inp, stat, stat, stat, stat},
            {true, true ,true, false, false}, {false, false, false, false, true});
    }
    {
        auto op = OprAttr::make("BatchNorm");
        auto&& attr = op->cast_final_safe<OprAttr>();
        Param param;
        param.fwd_mode = Param::FwdMode::TRAINING;
        attr.param.write_pod(param);
        OpDef::make_backward_graph(attr, {inp, stat, stat},
            {true, true ,true}, {false, false, true});
    }
}

TEST(TestImperative, OptimizedBackwardGraphBasic) {
    auto cn = CompNode::load("xpux");
    LogicalTensorDesc desc = {TensorLayout(dtype::Float32()), cn};
    HostTensorGenerator<> gen;
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto bg = OpDef::make_backward_graph(*op, {desc, desc}, {true, true}, {true});
    auto obg = OptimizedBackwardGraphResult(bg);
    ASSERT_EQ(obg.save_for_backward.size(), 4);
    ASSERT_FALSE(obg.save_for_backward[0]);
    ASSERT_FALSE(obg.save_for_backward[1]);
    ASSERT_FALSE(obg.save_for_backward[2]);

    auto a_hv = gen({42});
    auto b_hv = gen({5, 42});
    auto dc_hv = gen({5, 42});
    auto a_tn = Tensor::make(*a_hv);
    auto b_tn = Tensor::make(*b_hv);
    auto dc_tn = Tensor::make(*dc_hv);
    auto c_tn = OpDef::apply_on_physical_tensor(*op, {a_tn, b_tn})[0];

    auto backward_graph_inputs = prepare_backward_graph_inputs<SmallVector<TensorPtr>>(bg, {a_tn, b_tn}, {c_tn}, {dc_tn});
    auto grads = expand_grads(bg, OpDef::apply_on_physical_tensor(*bg.backward, backward_graph_inputs));

    auto precomp = OpDef::apply_on_physical_tensor(*obg.precomp, {a_tn, b_tn, c_tn});
    ASSERT_EQ(precomp.size(), 2);
    ASSERT_EQ(precomp[0]->shape().ndim, 1);
    ASSERT_LE(precomp[0]->shape()[0], 2);
    ASSERT_EQ(precomp[1]->shape().ndim, 1);
    ASSERT_LE(precomp[1]->shape()[0], 2);

    auto backward_inputs = prepare_optimized_backward_inputs<SmallVector<TensorPtr>>(obg, precomp, {a_tn, b_tn}, {c_tn}, {dc_tn});
    auto grads2 = expand_grads(obg, OpDef::apply_on_physical_tensor(*obg.backward, backward_inputs));

    ASSERT_EQ(grads2.size(), 2);
    MGB_ASSERT_TENSOR_EQ(grads[0]->get_value(), grads2[0]->get_value());
    MGB_ASSERT_TENSOR_EQ(grads[1]->get_value(), grads2[1]->get_value());
}
