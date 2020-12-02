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

std::shared_ptr<Tensor> reshape_to(Tensor* x, Tensor* s) {
    static auto op = Reshape::make();
    return python::apply(op, x, s)[0];
}

std::shared_ptr<Tensor> broadcast_to(Tensor* x, Tensor* s) {
    static auto op = Broadcast::make();
    return python::apply(op, x, s)[0];
}

std::shared_ptr<Tensor> make_tensor(CompNode cn, Tensor* shape, float v = 0) {
    HostTensorND scalar{cn, {{1}, dtype::Float32()}};
    scalar.ptr<float>()[0] = v;
    interpreter::Interpreter::Handle handle = interpreter_for_py->put(scalar, false);
    auto&& t = std::make_shared<Tensor>(handle);
    auto res = broadcast_to(t.get(), shape);
    return res;
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
            if (!grad) {
                return ret;
            }
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

apply_result_t reshape_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
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
        if (!grad) {
            return ret;
        }
        for (size_t i = 0; i < 2; ++i) {
            if (shapes[i]) {
                ret[i] = reshape_to(grad, shapes[i].get());
            }
        }
        return ret;
    });
    return apply(ctx);
}

apply_result_t subtensor_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto&& op = ctx.op->cast_final_safe<Subtensor>();
    auto&& grad_op = SetSubtensor::make(op.items);
    SmallVector<std::shared_ptr<Tensor>> inputs;
    if (input_requires_grad(ctx, 0)) {
        inputs.push_back(get_shape(ctx.args[0]));
        for (size_t i = 1; i < ctx.nargs; ++i) {
            inputs.push_back(ctx.args[i]->copy());
        }
    }
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs=std::move(inputs), grad_op_=std::move(grad_op)](BackwardContext&, Tensor*const* grads, size_t ngrads) {
        mgb_assert(ngrads == 1);
        Tensor* grad = grads[0];
        apply_result_t ret(1);
        if (grad && inputs[0]) {
            SmallVector<Tensor*> args_(inputs.size()+1);
            auto&& zeros = make_tensor(grad->comp_node(), inputs[0].get());
            args_[0] = zeros.get();
            args_[1] = grad;
            for (size_t i = 1; i < inputs.size(); ++i) {
                args_[i+1] = inputs[i].get();
            }
            ret[0] = python::apply(grad_op_, args_)[0];
        }
        return ret;
    });
    return apply(ctx);
}

apply_result_t indexingMultiAxisVec_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto&& op = ctx.op->cast_final_safe<IndexingMultiAxisVec>();
    auto&& grad_op = IndexingSetMultiAxisVec::make(op.items);
    SmallVector<std::shared_ptr<Tensor>> inputs;
    if (input_requires_grad(ctx, 0)) {
        inputs.push_back(get_shape(ctx.args[0]));
        for (size_t i = 1; i < ctx.nargs; ++i) {
            inputs.push_back(ctx.args[i]->copy());
        }
    }
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs=std::move(inputs), grad_op_=std::move(grad_op)](BackwardContext&, Tensor*const* grads, size_t ngrads) {
        mgb_assert(ngrads == 1);
        Tensor* grad = grads[0];
        apply_result_t ret(1);
        if (grad && inputs[0]) {
            SmallVector<Tensor*> args_(inputs.size()+1);
            auto&& zeros = make_tensor(grad->comp_node(), inputs[0].get());
            args_[0] = zeros.get();
            args_[1] = grad;
            for (size_t i = 1; i < inputs.size(); ++i) {
                args_[i+1] = inputs[i].get();
            }
            ret[0] = python::apply(grad_op_, args_)[0];
        }
        return ret;
    });
    return apply(ctx);
}

apply_result_t reduce_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto& op = ctx.op->cast_final_safe<Reduce>();
    if (op.mode == Reduce::Mode::SUM) {
        mgb_assert(ctx.nargs == 1);
        std::array<std::shared_ptr<Tensor>, 1> input_shapes;
        if (input_requires_grad(ctx, 0)) {
            input_shapes[0] = get_shape(ctx.args[0]);
        }
        maker.output_size(1).output_captured(0, false);
        maker.backward([shapes=std::move(input_shapes)](BackwardContext&, Tensor*const* grads, size_t ngrads) {
            mgb_assert(ngrads == 1);
            Tensor* grad = grads[0];
            apply_result_t ret(1);
            if (grad && shapes[0]) {
                ret[0] = broadcast_to(grad, shapes[0].get());
            }
            return ret;
        });
        return apply(ctx);
    }
    throw GradRuleFallback();
}

apply_result_t addAxis_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto&& op = ctx.op->cast_final_safe<AddAxis>();
    mgb_assert(ctx.nargs == 1);
    bool flag = input_requires_grad(ctx, 0);
    auto&& grad_op = RemoveAxis::make(op.axis);
    std::sort(grad_op->axis.begin(), grad_op->axis.end(), std::greater<int32_t>());
    maker.output_size(1).output_captured(0, false);
    maker.backward([grad_op_=std::move(grad_op), flag_=flag](BackwardContext&, Tensor*const* grads, size_t ngrads) {
        mgb_assert(ngrads == 1);
        Tensor* grad = grads[0];
        apply_result_t ret(1);
        if (grad && flag_) {
            ret[0] = python::apply(grad_op_, grad)[0];
        }
        return ret;
    });
    return apply(ctx);
}

apply_result_t removeAxis_grad_rule(ApplyContext& ctx, CustomBackward::Maker& maker) {
    auto&& op = ctx.op->cast_final_safe<RemoveAxis>();
    mgb_assert(ctx.nargs == 1);
    bool flag = input_requires_grad(ctx, 0);
    auto&& grad_op = AddAxis::make(op.axis);
    std::sort(grad_op->axis.begin(), grad_op->axis.end());
    maker.output_size(1).output_captured(0, false);
    maker.backward([grad_op_=std::move(grad_op), flag_=flag](BackwardContext&, Tensor*const* grads, size_t ngrads) {
        mgb_assert(ngrads == 1);
        Tensor* grad = grads[0];
        apply_result_t ret(1);
        if (grad && flag_) {
            ret[0] = python::apply(grad_op_, grad)[0];
        }
        return ret;
    });
    return apply(ctx);
}

struct Init {
    Init() {
        auto& reg = grad_rule_registry();
        reg.emplace(Elemwise::typeinfo(), elemwise_grad_rule);
        reg.emplace(Reshape::typeinfo(), reshape_grad_rule);
        reg.emplace(Subtensor::typeinfo(), subtensor_grad_rule);
        reg.emplace(IndexingMultiAxisVec::typeinfo(), indexingMultiAxisVec_grad_rule);
        reg.emplace(Reduce::typeinfo(), reduce_grad_rule);
        reg.emplace(AddAxis::typeinfo(), addAxis_grad_rule);
        reg.emplace(RemoveAxis::typeinfo(), removeAxis_grad_rule);
    }
} _;

} // namespace
} // namespace mgb::imperative::python
