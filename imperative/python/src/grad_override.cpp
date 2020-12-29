/**
 * \file imperative/python/src/grad_override.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb::imperative::python {
namespace {

std::shared_ptr<Tensor> get_shape(Tensor* x) {
    static auto op = GetVarShape::make();
    return python::apply(op, x)[0];
}

std::shared_ptr<Tensor> reduce_to(Tensor* x, Tensor* s) {
    static auto op = Reduce::make();
    return python::apply(op, x, s)[0];
}

apply_result_t elemwise_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto& op = ctx.op->cast_final_safe<Elemwise>();
    if (op.mode == Elemwise::Mode::ADD) {
        mgb_assert(ctx.nargs == 2);
        std::array<std::shared_ptr<Tensor>, 2> input_shapes;
        for (size_t i = 0; i < 2; ++i) {
            if (input_requires_grad(ctx, i)) {
                input_shapes[i] = get_shape(ctx.args[i]);
            }
        }
        maker.output_size(1).output_captured(0, false);
        maker.backward([shapes=std::move(input_shapes)](BackwardContext&, Tensor*const* grads, size_t ngrads) {
            mgb_assert(ngrads == 1);
            Tensor* grad = grads[0];
            apply_result_t ret(2);
            for (size_t i = 0; i < 2; ++i) {
                if (shapes[i]) {
                    ret[i] = reduce_to(grad, shapes[i].get());
                }
            }
            return ret;
        });
        return apply(ctx);
    }
    throw GradRuleFallback();
}

struct Init {
    Init() {
        auto& reg = grad_rule_registry();
        reg.emplace(Elemwise::typeinfo(), elemwise_grad_rule);
    }
} _;

} // namespace
} // namespace mgb::imperative::python
