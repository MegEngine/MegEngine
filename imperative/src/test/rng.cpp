/**
 * \file imperative/src/test/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"
#include "megbrain/imperative/ops/rng.h"

using namespace mgb;
using namespace imperative;
using namespace imperative::rng;

template<typename Op, typename ...Args>
void check_rng_basic(Args&& ...args) {
    for (auto&& tshape: {
        TensorShape{2, 3, 4, 5},
        {3, 4, 5, 6},
        {2333}})
    for (auto&& cn: {
        CompNode::load("xpu0"),
        CompNode::load("xpu1")})
    {
        Handle h = new_handle(cn, 123);
        auto op = Op::make(std::forward<Args>(args)..., h);
        DeviceTensorND tshape_dev;
        cg::copy_shape_to_tensor_value(tshape_dev, tshape);
        SmallVector<TensorPtr> inputs = {Tensor::make(tshape_dev)};
        auto outputs = OpDef::apply_on_physical_tensor(*op, inputs);
        ASSERT_TRUE(outputs[0]->layout().eq_shape(tshape));
        ASSERT_TRUE(cn == outputs[0]->comp_node());
        // sync before delete handle
        for (auto&& p: outputs) {
            p->get_value();
        }
        delete_handle(h);
    }
}

TEST(TestImperative, UniformRNGBasic) {
    check_rng_basic<UniformRNG>(123);
}

TEST(TestImperative, GaussianRNGBasic) {
    check_rng_basic<GaussianRNG>(123, 2.f, 3.f);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
