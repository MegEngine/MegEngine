#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/transformations/grad.h"

namespace mgb::imperative::python {

class CustomGradMaker {
    bool output_size_set = false, input_has_grad_initialized = false;
    CustomBackward& target;
    size_t nr_inputs;
    void init_input_has_grad() {
        if (!input_has_grad_initialized) {
            input_has_grad_initialized = true;
            target.m_input_has_grad.resize(nr_inputs, true);
        }
    }

public:
    CustomGradMaker(CustomBackward& target, size_t nr_inputs)
            : target(target), nr_inputs(nr_inputs) {}

    CustomGradMaker& backward(CustomBackward::BackwardFn f) {
        mgb_assert(!target.m_backward);
        target.m_backward = f;
        return *this;
    }
    // mandatory
    CustomGradMaker& output_size(size_t sz) {
        mgb_assert(!output_size_set);
        output_size_set = true;
        target.m_output_attrs.resize(sz);
        return *this;
    }
    // optional, defaults to all true
    CustomGradMaker& input_has_grad(size_t i, bool v) {
        init_input_has_grad();
        target.m_input_has_grad.at(i) = v;
        return *this;
    }
    // optional, defaults to all true
    CustomGradMaker& output_requires_grad(size_t i, bool v) {
        target.m_output_attrs.at(i).requires_grad = v;
        return *this;
    }
    // optional, defaults to all true
    CustomGradMaker& output_captured(size_t i, bool v) {
        target.m_output_attrs.at(i).captured = v;
        return *this;
    }
    void finalize() {
        mgb_assert(output_size_set);
        init_input_has_grad();
    }
};

namespace {

ValueRef get_shape(ValueRef x) {
    static auto op = GetVarShape::make();
    return imperative::apply(*op, x)[0];
}

ValueRef reduce_to(ValueRef x, ValueRef s) {
    static auto op = Reduce::make();
    return imperative::apply(*op, x, s)[0];
}

ValueRef reshape_to(ValueRef x, ValueRef s) {
    static auto op = Reshape::make();
    return imperative::apply(*op, x, s)[0];
}

ValueRef broadcast_to(ValueRef x, ValueRef s) {
    static auto op = Broadcast::make();
    return imperative::apply(*op, x, s)[0];
}

ValueRef make_empty_tensor(
        CompNodeValue::ref_t device, ValueRef shape, DTypeValue::ref_t dtype) {
    HostTensorStorage storage(*device);
    storage.ensure_size(dtype->size());
    std::memset(storage.ptr(), 0, dtype->size());
    auto t = imperative::apply(
            CreateTensor(CreateTensor::Unique, *device, *dtype, ValueShape()),
            HostStorage::make(storage))[0];
    auto res = broadcast_to(t, shape);
    return res;
}

std::optional<ValueRefList> matrix_mul_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& matmul = op.cast_final_safe<MatrixMul>();
    size_t dimA = matmul.dimA;
    size_t dimB = matmul.dimB;
    auto&& param = matmul.param();
    auto&& policy = matmul.policy();
    mgb_assert(inputs.size() == 2);
    std::array<ValueRef, 2> inps, input_shapes;
    for (size_t i = 0; i < 2; ++i) {
        if (inputs_require_grad[i ^ 1]) {
            inps[i] = inputs[i];
            input_shapes[i] = get_shape(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inps_ = std::move(inps), input_shapes_ = std::move(input_shapes),
                    param, policy, dimA, dimB](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(2);
        if (!grad) {
            return ret;
        }
        size_t dimG = std::max(dimA, dimB);
        if (inps_[1]) {
            if (param.transposeA) {
                // A^T(2) @ B(2) = G(2), A'(2) = B'(2) @ G'^T(2) -> MatrixMul
                auto&& grad_op = MatrixMul::make(
                        param.transposeB, true, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimB, dimG);
                ret[0] = imperative::apply(*grad_op, inps_[1], grad)[0];
            } else {
                // A(>=2) @ B(2) = G(>=2), A'(>=2) = G'(>=2) @ B(2) -> MatrixMul
                auto&& grad_op = MatrixMul::make(
                        false, !param.transposeB, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimG, dimB);
                ret[0] = imperative::apply(*grad_op, grad, inps_[1])[0];
            }
        }
        if (inps_[0]) {
            if (param.transposeB) {
                // A(>=2) @ B^T(2) = G(>=2), B'(2) = G'^T(>=2) @ A(>=2) -> MatrixMul
                // (specialized)
                auto&& grad_op = MatrixMul::make(
                        true, param.transposeA, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimG, dimA);
                ret[1] = imperative::apply(*grad_op, grad, inps_[0])[0];
            } else {
                // A(>=2) @ B(2) = G(>=2), B'(2) = G'(>=2) @ A(>=2) -> MatrixMul
                // (specialized)
                auto&& grad_op = MatrixMul::make(
                        !param.transposeA, false, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimA, dimG);
                ret[1] = imperative::apply(*grad_op, inps_[0], grad)[0];
            }
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> batched_matrix_mul_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& bmm = op.cast_final_safe<BatchedMatrixMul>();
    size_t dimA = bmm.dimA;
    size_t dimB = bmm.dimB;
    auto&& param = bmm.param();
    auto&& policy = bmm.policy();
    mgb_assert(inputs.size() == 2);
    std::array<ValueRef, 2> inps, input_shapes;
    for (size_t i = 0; i < 2; ++i) {
        if (inputs_require_grad[i ^ 1]) {
            inps[i] = inputs[i];
            input_shapes[i] = get_shape(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inps_ = std::move(inps), input_shapes_ = std::move(input_shapes),
                    param, policy, dimA, dimB](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(2);
        if (!grad) {
            return ret;
        }
        size_t dimG = std::max(dimA, dimB);
        if (inps_[1]) {
            if (param.transposeA) {
                auto&& grad_op = BatchedMatrixMul::make(
                        param.transposeB, true, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimB, dimG);
                ret[0] = imperative::apply(*grad_op, inps_[1], grad)[0];
            } else {
                auto&& grad_op = BatchedMatrixMul::make(
                        false, !param.transposeB, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimG, dimB);
                ret[0] = imperative::apply(*grad_op, grad, inps_[1])[0];
            }
            if (dimG != dimA) {
                ret[0] = reduce_to(ret[0], input_shapes_[0]);
            }
        }
        if (inps_[0]) {
            if (param.transposeB) {
                auto&& grad_op = BatchedMatrixMul::make(
                        true, param.transposeA, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimG, dimA);
                ret[1] = imperative::apply(*grad_op, grad, inps_[0])[0];
            } else {
                auto&& grad_op = BatchedMatrixMul::make(
                        !param.transposeA, false, param.compute_mode, param.format,
                        policy.strategy, policy.workspace_limit, dimA, dimG);
                ret[1] = imperative::apply(*grad_op, inps_[0], grad)[0];
            }
            if (dimG != dimB) {
                ret[1] = reduce_to(ret[1], input_shapes_[1]);
            }
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> elemwise_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto& elemwise = op.cast_final_safe<Elemwise>();
    if (elemwise.mode != Elemwise::Mode::ADD) {
        return {};
    }
    mgb_assert(inputs.size() == 2);
    std::array<ValueRef, 2> input_shapes;
    for (size_t i = 0; i < 2; ++i) {
        if (inputs_require_grad[i]) {
            input_shapes[i] = get_shape(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([shapes = std::move(input_shapes)](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(2);
        if (!grad) {
            return ret;
        }
        for (size_t i = 0; i < 2; ++i) {
            if (shapes[i]) {
                ret[i] = reduce_to(grad, shapes[i]);
            }
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> reshape_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    mgb_assert(inputs.size() == 1 || inputs.size() == 2);
    size_t nr_inp = inputs.size();
    std::array<ValueRef, 2> input_shapes;
    for (size_t i = 0; i < nr_inp; ++i) {
        if (inputs_require_grad[i]) {
            input_shapes[i] = get_shape(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([shapes = std::move(input_shapes), nr_inp](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(nr_inp);
        if (!grad) {
            return ret;
        }
        for (size_t i = 0; i < nr_inp; ++i) {
            if (shapes[i]) {
                ret[i] = reshape_to(grad, shapes[i]);
            }
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> broadcast_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    mgb_assert(inputs.size() == 1 || inputs.size() == 2);
    size_t nr_inp = inputs.size();
    std::array<ValueRef, 2> input_shapes;
    for (size_t i = 0; i < nr_inp; ++i) {
        if (inputs_require_grad[i]) {
            input_shapes[i] = get_shape(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([shapes = std::move(input_shapes), nr_inp](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(nr_inp);
        if (!grad) {
            return ret;
        }
        for (size_t i = 0; i < nr_inp; ++i) {
            if (shapes[i]) {
                ret[i] = reduce_to(grad, shapes[i]);
            }
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> subtensor_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& subtensor = op.cast_final_safe<Subtensor>();
    auto&& grad_op = SetSubtensor::make(subtensor.items);
    SmallVector<ValueRef> inputs2;
    if (inputs_require_grad[0]) {
        inputs2.push_back(get_shape(inputs[0]));
        for (size_t i = 1; i < inputs.size(); ++i) {
            inputs2.push_back(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs = std::move(inputs2),
                    grad_op_ = std::move(grad_op)](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && inputs[0]) {
            ValueRefList args_(inputs.size() + 1);
            auto&& zeros = make_empty_tensor(grad.device(), inputs[0], grad.dtype());
            args_[0] = zeros;
            args_[1] = grad;
            for (size_t i = 1; i < inputs.size(); ++i) {
                args_[i + 1] = inputs[i];
            }
            ret[0] = imperative::apply(ApplyOp(*grad_op_), args_)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> indexingMultiAxisVec_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& indexingMultiAxisVec = op.cast_final_safe<IndexingMultiAxisVec>();
    auto&& grad_op = IndexingIncrMultiAxisVec::make(indexingMultiAxisVec.items);
    SmallVector<ValueRef> inputs2;
    if (inputs_require_grad[0]) {
        inputs2.push_back(get_shape(inputs[0]));
        for (size_t i = 1; i < inputs.size(); ++i) {
            inputs2.push_back(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs = std::move(inputs2),
                    grad_op_ = std::move(grad_op)](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && inputs[0]) {
            ValueRefList args_(inputs.size() + 1);
            auto&& zeros = make_empty_tensor(grad.device(), inputs[0], grad.dtype());
            args_[0] = zeros;
            args_[1] = grad;
            for (size_t i = 1; i < inputs.size(); ++i) {
                args_[i + 1] = inputs[i];
            }
            ret[0] = imperative::apply(ApplyOp(*grad_op_), args_)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> reduce_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto& reduce = op.cast_final_safe<Reduce>();
    if (reduce.mode != Reduce::Mode::SUM) {
        return {};
    }
    auto axis = reduce.axis;
    if (inputs.size() != 1 || axis == INT_MAX) {
        return {};
    }
    std::array<ValueRef, 1> input_shapes;
    if (inputs_require_grad[0]) {
        input_shapes[0] = get_shape(inputs[0]);
    }
    if (axis < 0) {
        axis = (*inputs[0].shape()).ndim + axis;
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    auto keepdim = reduce.keepdim || axis == INT_MAX;
    maker.output_size(1).output_captured(0, false);
    maker.backward(
            [shapes = std::move(input_shapes), axis, keepdim](Span<ValueRef> grads) {
                mgb_assert(grads.size() == 1);
                ValueRef grad = grads[0];
                if (!keepdim) {
                    auto&& grad_op = AddAxis::make(std::vector<int32_t>({axis}));
                    grad = imperative::apply(*grad_op, grad)[0];
                }
                SmallVector<ValueRef> ret(1);
                if (grad && shapes[0]) {
                    ret[0] = broadcast_to(grad, shapes[0]);
                }
                return ret;
            });
    maker.finalize();
    return imperative::apply(ApplyOp(op), inputs);
}

std::optional<ValueRefList> addAxis_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& addAxis = op.cast_final_safe<AddAxis>();
    mgb_assert(inputs.size() == 1);
    bool flag = inputs_require_grad[0];
    auto&& grad_op = RemoveAxis::make(addAxis.axis);
    std::sort(grad_op->axis.begin(), grad_op->axis.end(), std::greater<int32_t>());
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([grad_op_ = std::move(grad_op), flag_ = flag](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && flag_) {
            ret[0] = imperative::apply(*grad_op_, grad)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

std::optional<ValueRefList> removeAxis_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& removeAxis = op.cast_final_safe<RemoveAxis>();
    mgb_assert(inputs.size() == 1);
    bool flag = inputs_require_grad[0];
    auto&& grad_op = AddAxis::make(removeAxis.axis);
    std::sort(grad_op->axis.begin(), grad_op->axis.end());
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([grad_op_ = std::move(grad_op), flag_ = flag](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && flag_) {
            ret[0] = imperative::apply(*grad_op_, grad)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

std::optional<ValueRefList> pixelShuffle_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& pixelShuffle = op.cast_final_safe<PixelShuffle>();
    mgb_assert(inputs.size() == 1);
    bool flag = inputs_require_grad[0];
    auto&& grad_op = PixelShuffleBackward::make(pixelShuffle.factor);
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([grad_op_ = std::move(grad_op), flag_ = flag](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && flag_) {
            ret[0] = imperative::apply(*grad_op_, grad)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

std::optional<ValueRefList> indexing_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& indexing = op.cast_final_safe<IndexingOneHot>();
    mgb_assert(inputs.size() == 2);
    bool flag = inputs_require_grad[0];
    auto&& grad_op = IndexingSetOneHot::make(indexing.axis, indexing.ndim);
    SmallVector<ValueRef> inputs2;
    if (flag) {
        inputs2.push_back(get_shape(inputs[0]));
        for (size_t i = 1; i < inputs.size(); ++i) {
            inputs2.push_back(inputs[i]);
        }
    }
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs = std::move(inputs2),
                    grad_op_ = std::move(grad_op)](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad && inputs[0]) {
            ValueRefList args_(inputs.size() + 1);
            auto&& zeros = make_empty_tensor(grad.device(), inputs[0], grad.dtype());
            args_[0] = zeros;
            args_[1] = inputs[1];
            args_[2] = grads[0];
            ret[0] = imperative::apply(*grad_op_, args_)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

std::optional<ValueRefList> indexing_set_one_hot_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    auto&& indexingSetOneHot = op.cast_final_safe<IndexingSetOneHot>();
    mgb_assert(inputs.size() == 3);
    SmallVector<ValueRef> inputs2;
    inputs2.push_back(get_shape(inputs[0]));
    inputs2.push_back(inputs[1]);
    inputs2.push_back(get_shape(inputs[2]));
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([inputs = std::move(inputs2),
                    &indexingSetOneHot](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(3);
        if (!grad) {
            return ret;
        }
        if (inputs[0]) {
            auto&& grad_op = IndexingSetOneHot::make(
                    indexingSetOneHot.axis, indexingSetOneHot.ndim);
            ValueRefList args_(inputs.size());
            auto&& zeros = make_empty_tensor(grad.device(), inputs[2], grad.dtype());
            args_[0] = grads[0];
            args_[1] = inputs[1];
            args_[2] = zeros;
            ret[0] = imperative::apply(*grad_op, args_)[0];
        }
        if (inputs[2]) {
            auto&& grad_op = IndexingOneHot::make(
                    indexingSetOneHot.axis, indexingSetOneHot.ndim);
            ValueRefList args_(inputs.size() - 1);
            args_[0] = grads[0];
            args_[1] = inputs[1];
            ret[2] = imperative::apply(*grad_op, args_)[0];
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

std::optional<ValueRefList> fastpathcopy_grad_rule(
        const OpDef& op, Span<ValueRef> inputs, Span<bool> inputs_require_grad,
        CustomBackward& backward) {
    mgb_assert(inputs.size() == 1);
    auto maker = CustomGradMaker(backward, inputs.size());
    maker.output_size(1).output_captured(0, false);
    maker.backward([](Span<ValueRef> grads) {
        mgb_assert(grads.size() == 1);
        ValueRef grad = grads[0];
        SmallVector<ValueRef> ret(1);
        if (grad) {
            ret[0] = grad;
        }
        return ret;
    });
    maker.finalize();
    return imperative::apply(op, inputs);
}

struct Init {
    Init() {
        CustomBackward::register_grad_rule(Elemwise::typeinfo(), elemwise_grad_rule);
        CustomBackward::register_grad_rule(Reshape::typeinfo(), reshape_grad_rule);
        CustomBackward::register_grad_rule(Broadcast::typeinfo(), broadcast_grad_rule);
        CustomBackward::register_grad_rule(Subtensor::typeinfo(), subtensor_grad_rule);
        CustomBackward::register_grad_rule(
                IndexingMultiAxisVec::typeinfo(), indexingMultiAxisVec_grad_rule);
        CustomBackward::register_grad_rule(Reduce::typeinfo(), reduce_grad_rule);
        CustomBackward::register_grad_rule(AddAxis::typeinfo(), addAxis_grad_rule);
        CustomBackward::register_grad_rule(
                RemoveAxis::typeinfo(), removeAxis_grad_rule);
        CustomBackward::register_grad_rule(
                IndexingOneHot::typeinfo(), indexing_grad_rule);
        CustomBackward::register_grad_rule(
                IndexingSetOneHot::typeinfo(), indexing_set_one_hot_grad_rule);
        CustomBackward::register_grad_rule(
                FastpathCopy::typeinfo(), fastpathcopy_grad_rule);
        CustomBackward::register_grad_rule(
                PixelShuffle::typeinfo(), pixelShuffle_grad_rule);
        CustomBackward::register_grad_rule(MatrixMul::typeinfo(), matrix_mul_grad_rule);
        CustomBackward::register_grad_rule(
                BatchedMatrixMul::typeinfo(), batched_matrix_mul_grad_rule);
    }
} _;

}  // namespace
}  // namespace mgb::imperative::python
