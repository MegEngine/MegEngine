#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"

using namespace megdnn;

namespace mgb::imperative {

namespace pixel_shuffle {

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<PixelShuffle>();
    auto&& src = inputs[0];
    auto&& layout = src->layout();
    mgb_assert(
            layout.ndim >= 3,
            "the input dimension of pixel_shuffle should be larger than or equal to 3");
    size_t idx = layout.ndim - 3;
    mgb_assert(
            layout[idx] % (op.factor * op.factor) == 0,
            "the -3 dimension should be divided by (upscale_factor ** 2)");
    TensorLayout tlayout;
    TensorShape tshp;  // {N, C, r, r, H, W}
    TensorShape vshp;  // {..., C, Hr, Wr}
    tshp.ndim = 6;
    vshp.ndim = layout.ndim;
    tshp[0] = 1;
    for (size_t i = 0; i < idx; ++i) {
        tshp[0] *= layout[i];
        vshp[i] = layout[i];
    }
    tshp[1] = layout[idx] / (op.factor * op.factor);
    tshp[2] = tshp[3] = op.factor;
    tshp[4] = layout[idx + 1];
    tshp[5] = layout[idx + 2];
    vshp[idx] = tshp[1];
    vshp[idx + 1] = layout[idx + 1] * op.factor;
    vshp[idx + 2] = layout[idx + 2] * op.factor;
    tlayout = layout.reshape(tshp).dimshuffle({0, 1, 4, 2, 5, 3});
    TensorPtr out = Tensor::make(src->blob(), src->offset(), tlayout);
    out->to_contiguous_inplace();  // relayout
    tlayout = out->layout().reshape(vshp);
    return {Tensor::make(out->blob(), out->offset(), tlayout)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<PixelShuffle>();
    mgb_assert(op.factor > 0, "upscale_factor should be larger than 0");
    auto&& src = inputs[0];
    if (src.layout.ndim == 0) {
        return {{{TensorLayout(src.layout.dtype), src.comp_node}}, false};
    }
    mgb_assert(
            src.layout.ndim >= 3,
            "the input dimension of pixel_shuffle should be larger than or equal to 3");
    size_t idx = src.layout.ndim - 3;
    mgb_assert(
            src.layout[idx] % (op.factor * op.factor) == 0,
            "the -3 dimension should be divided by (upscale_factor ** 2)");
    TensorShape tshp;
    tshp.ndim = src.layout.ndim;
    for (size_t i = 0; i < idx; ++i) {
        tshp[i] = src.layout[i];
    }
    tshp[idx] = src.layout[idx] / (op.factor * op.factor);
    tshp[idx + 1] = src.layout[idx + 1] * op.factor;
    tshp[idx + 2] = src.layout[idx + 2] * op.factor;
    return {{{TensorLayout(tshp, src.layout.dtype), src.comp_node}}, true};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(PixelShuffle, PixelShuffle)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace pixel_shuffle

namespace pixel_shuffle_backward {

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<PixelShuffleBackward>();
    auto&& src = inputs[0];
    auto&& layout = src->layout();
    size_t idx = layout.ndim - 3;
    TensorLayout tlayout;
    TensorShape tshp;  // {N, C, H, r, W, r}
    TensorShape vshp;  // {..., Cr^2, H, W}
    tshp.ndim = 6;
    vshp.ndim = layout.ndim;
    tshp[0] = 1;
    for (size_t i = 0; i < idx; ++i) {
        tshp[0] *= layout[i];
        vshp[i] = layout[i];
    }
    tshp[1] = layout[idx];
    tshp[3] = tshp[5] = op.factor;
    tshp[2] = layout[idx + 1] / op.factor;
    tshp[4] = layout[idx + 2] / op.factor;
    vshp[idx] = tshp[1] * op.factor * op.factor;
    vshp[idx + 1] = tshp[2];
    vshp[idx + 2] = tshp[4];
    tlayout = layout.reshape(tshp).dimshuffle({0, 1, 3, 5, 2, 4});
    TensorPtr out = Tensor::make(src->blob(), src->offset(), tlayout);
    out->to_contiguous_inplace();  // relayout
    tlayout = out->layout().reshape(vshp);
    return {Tensor::make(out->blob(), out->offset(), tlayout)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<PixelShuffleBackward>();
    auto&& src = inputs[0];
    if (src.layout.ndim == 0) {
        return {{{TensorLayout(src.layout.dtype), src.comp_node}}, false};
    }
    size_t idx = src.layout.ndim - 3;
    TensorShape tshp;
    tshp.ndim = src.layout.ndim;
    for (size_t i = 0; i < idx; ++i) {
        tshp[i] = src.layout[i];
    }
    tshp[idx] = src.layout[idx] * op.factor * op.factor;
    tshp[idx + 1] = src.layout[idx + 1] / op.factor;
    tshp[idx + 2] = src.layout[idx + 2] / op.factor;
    return {{{TensorLayout(tshp, src.layout.dtype), src.comp_node}}, true};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(PixelShuffleBackward, PixelShuffleBackward)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace pixel_shuffle_backward

}  // namespace mgb::imperative
