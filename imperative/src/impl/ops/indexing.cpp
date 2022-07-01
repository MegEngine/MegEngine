#include "../dnn_op_helper.h"
#include "megbrain/imperative/ops/autogen.h"

#include "../op_trait.h"

#include "megbrain/opr/indexing.h"
#include "megdnn/oprs/general.h"

namespace mgb {
namespace imperative {

namespace {
namespace indexing_one_hot {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto&& op = def.cast_final_safe<IndexingOneHot>();
    mgb_assert(input_descs.size() == 2, "IndexingOneHot expects two inputs");
    auto comp_node = input_descs[0].comp_node;
    TensorLayout src = input_descs[0].layout, index = input_descs[1].layout;

    mgb_assert(index.dtype == dtype::Int32(), "index dtype must be int32");

    if (!src.ndim) {
        return {{{{{}, src.dtype}, comp_node}}, false};
    }

    mgb_assert(src.ndim >= 2, "src ndim must be at least 2");
    mgb_assert(src.is_contiguous(), "src should be contiguous");
    mgb_assert(
            -static_cast<int>(src.ndim) <= op.axis &&
                    op.axis < static_cast<int>(src.ndim),
            "axis %d not exists in src", op.axis);
    int real_axis = static_cast<int>(op.axis);
    if (real_axis < 0) {
        real_axis += static_cast<int>(src.ndim);
    }
    TensorLayout dst = src;
    dst.shape[real_axis] = 1;
    dst.init_contiguous_stride();

    if (!index.ndim) {
        return {{{dst, comp_node}}, false};
    }

    mgb_assert(index.is_contiguous(), "index should be all contiguous");
    mgb_assert(
            index.eq_shape(src.remove_axis(real_axis)),
            "index shape doesn't match src");
    return {{{dst, comp_node}}, true};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<IndexingOneHot>();
    mgb_assert(inputs.size() == 2);
    int real_axis = static_cast<int>(op.axis);
    if (real_axis < 0) {
        real_axis += static_cast<int>(op.ndim);
    }
    OperatorNodeConfig config{op.make_name()};
    return opr::IndexingOneHot::make(inputs[0], inputs[1], real_axis, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<IndexingOneHot>();
    auto&& inp = inputs[0];
    auto&& index = inputs[1];
    auto&& layout = inp->layout();
    auto&& index_layout = index->layout();
    int real_axis = static_cast<int>(op.axis);
    if (real_axis < 0) {
        real_axis += static_cast<int>(layout.ndim);
    }
    mgb_assert(
            0 <= real_axis && real_axis < static_cast<int>(layout.ndim),
            "Dimension out of range (expected to be in range of [%d, %d], but got %d)",
            0, static_cast<int>(layout.ndim) - 1, op.axis);
    DnnOprCaller<megdnn::IndexingOneHot> dnn_op(inp->comp_node(), real_axis);
    auto tlayout = dnn_op.deduce_layout(layout, index_layout);
    auto out = Tensor::make(tlayout, inp->comp_node());
    dnn_op.exec_with_ws(inp, index, out);
    return {out};
}

OP_TRAIT_REG(IndexingOneHot, IndexingOneHot)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace indexing_one_hot

namespace indexing_set_one_hot {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    mgb_assert(input_descs.size() == 3, "IndexingSetOneHot expects three inputs");
    auto comp_node = input_descs[0].comp_node;
    auto&& src = input_descs[0].layout;
    auto&& index = input_descs[1].layout;
    mgb_assert(index.dtype == dtype::Int32(), "index dtype must be int32");
    if (!src.ndim) {
        return {{{{{}, src.dtype}, comp_node}}, false};
    }
    mgb_assert(src.is_contiguous(), "src should be contiguous");
    return {{{src, comp_node}}, true};
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const IndexingSetOneHot&>(def);
    mgb_assert(inputs.size() == 3);
    int real_axis = static_cast<int>(op.axis);
    if (real_axis < 0) {
        real_axis += static_cast<int>(op.ndim);
    }
    OperatorNodeConfig config{op.make_name()};
    return opr::IndexingSetOneHot::make(
            inputs[0], inputs[1], inputs[2], real_axis, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<IndexingSetOneHot>();
    auto&& inp = inputs[0];
    auto&& index = inputs[1];
    auto&& sub = inputs[2];
    TensorLayout layout = inp->layout();
    mgb_assert(layout.is_contiguous());
    int real_axis = static_cast<int>(op.axis);
    if (real_axis < 0) {
        real_axis += static_cast<int>(layout.ndim);
    }
    DnnOprCaller<megdnn::IndexingSetOneHot> dnn_op(inp->comp_node(), real_axis);
    TensorPtr out = Tensor::make(layout, inp->comp_node());
    out->dev_tensor().copy_from_fixlayout(inp->dev_tensor());
    dnn_op.exec_with_ws(out, index, sub);
    return {out};
}

OP_TRAIT_REG(IndexingSetOneHot, IndexingSetOneHot)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace indexing_set_one_hot

}  // anonymous namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
