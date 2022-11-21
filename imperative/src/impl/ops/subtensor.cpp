#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/indexing_helper.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/tensor.h"

#include "../algo_chooser.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

using namespace mgb::opr::indexing;
namespace mgb::imperative {

namespace {
namespace subtensor {

auto get_index(
        const VarNodeArray& inputs,
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& mask,
        const std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>>& slice) {
    size_t length = mask.size();
    auto graph = inputs[0]->owner_graph();
    auto comp_node = inputs[0]->comp_node();
    opr::Subtensor::IndexDesc ret(length);
    auto immutable_node = [&](int val) {
        DTypeScalar scalar = DTypeScalar(static_cast<megdnn::dt_int32>(val));
        return opr::ImmutableTensor::make(*graph, scalar, {comp_node});
    };
    for (size_t i = 0; i < length; ++i) {
        auto&& [axis, b_flag, e_flag, s_flag, idx_flag] = mask[i];
        auto&& [b_val, e_val, s_val, ax_val] = slice[i];
        ret[i].axis = axis;
        if (idx_flag) {
            ret[i].idx = immutable_node(ax_val);
        } else {
            if (b_flag) {
                ret[i].begin = immutable_node(b_val);
            }
            if (e_flag) {
                ret[i].end = immutable_node(e_val);
            }
            if (s_flag) {
                ret[i].step = immutable_node(s_val);
            }
        }
    }
    return ret;
}

auto origin_get_index(
        const VarNodeArray& inputs, size_t vidx,
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& mask) {
    size_t length = mask.size();
    opr::Subtensor::IndexDesc ret(length);
    for (size_t i = 0; i < length; ++i) {
        auto&& [axis, begin, end, step, idx] = mask[i];
        ret[i].axis = axis;
        if (idx) {
            ret[i].idx = inputs[vidx++];
        } else {
            mgb_assert(begin || end || step);
            if (begin)
                ret[i].begin = inputs[vidx++];
            if (end)
                ret[i].end = inputs[vidx++];
            if (step)
                ret[i].step = inputs[vidx++];
        }
    }
    mgb_assert(vidx == inputs.size());
    return ret;
}

TensorLayout deduce_layout(
        TensorLayout src, std::vector<std::tuple<int8_t, bool, bool, bool, bool>> items,
        std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t>> slice_items) {
    auto mod_size = [](int v, int size_ax) -> int {
        if (size_ax == 0)
            return 0;
        return v < 0 ? v + size_ax : v;
    };
#define CHECK(cond) \
    mgb_assert(cond, "index out of bound: layout=%s", src.to_string().c_str())

    for (int i = items.size() - 1; i >= 0; i--) {
        auto&& [axis, b_flag, e_flag, s_flag, idx_flag] = items[i];
        auto&& [b_val, e_val, s_val, ax_val] = slice_items[i];
        int shape_axis = src.shape[axis];
        int slice_step = s_val == INT_MAX ? 1 : s_val;
        int slice_start = b_val == INT_MIN ? 0 : b_val;
        int slice_stop = e_val == INT_MAX ? shape_axis : e_val;
        if (slice_step > 0) {
            slice_start = mod_size(slice_start, shape_axis);
            slice_stop = mod_size(slice_stop, shape_axis);
            slice_stop = std::min(slice_stop, shape_axis);
            slice_start = std::min(slice_start, slice_stop);
            CHECK(slice_start >= 0 && slice_stop >= slice_start &&
                  slice_stop <= shape_axis);
        } else {
            slice_start = s_val == INT_MIN ? shape_axis - 1 : b_val;
            slice_start = mod_size(slice_start, shape_axis);
            slice_stop = e_val == INT_MAX ? -1 : mod_size(e_val, shape_axis);
            slice_start = std::min(slice_start, std::max(shape_axis - 1, 0));
            slice_stop = std::min(slice_stop, slice_start);
            CHECK(slice_step < 0 && slice_start >= 0 && slice_stop <= slice_start &&
                  slice_start < shape_axis && slice_stop >= -1);
        }
        int abs_step = std::abs(slice_step);
        if (axis < 0) {
            axis = axis + src.ndim;
        };

        if (idx_flag == true) {
            if (src.ndim == 1) {
                src.shape[0] = 1;
            } else {
                src.remove_axis_inplace(axis);
            }
        } else {
            src.shape[axis] =
                    (std::abs(slice_stop - slice_start) + abs_step - 1) / abs_step;
            src.stride[axis] *= slice_step;
        }
    }
    return src;
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Subtensor&>(def);
    OperatorNodeConfig config{op.make_name()};
    if (inputs.size() > 1) {
        return opr::Subtensor::make(
                inputs[0], origin_get_index(inputs, 1, op.items), config);
    } else {
        return opr::Subtensor::make(
                inputs[0], get_index(inputs, op.items, op.slice_items), config);
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    if (inputs.size() >= 2) {
        return proxy_graph_detail::infer_output_attrs_fallible(def, inputs);
    }
    auto&& inp = inputs[0];
    auto& inp_cn = inp.comp_node;
    if (inp.layout.ndim == 0) {
        return {{{TensorLayout{inp.layout.dtype}, inp_cn, {}}}, false};
    }
    auto&& op = static_cast<const Subtensor&>(def);

    auto items = op.items;
    auto slice_itmes = op.slice_items;
    TensorLayout out_layout = deduce_layout(inp.layout, items, slice_itmes);

    return {{{out_layout, inp_cn, {}}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    CompNode cn = inputs[0]->comp_node();
    auto&& layout = inputs[0]->layout();
    auto&& op = static_cast<const Subtensor&>(def);

    if (inputs.size() > 1) {
        return proxy_graph_detail::apply_on_physical_tensor(
                def, inputs, output_descs, validated);
    }
    auto&& src = inputs[0];
    auto slice_items = op.slice_items;
    auto items = op.items;
    TensorLayout res_layout = deduce_layout(layout, items, slice_items);
    if (res_layout.is_empty()) {
        return {Tensor::make(res_layout, cn)};
    }
    size_t offset = 0;
    size_t dtype_size = layout.dtype.size();
    TensorPtr tensor = src;
    for (int i = items.size() - 1; i >= 0; i--) {
        auto&& [axis, b_flag, e_flag, s_flag, idx_flag] = items[i];
        auto&& [b_val, e_val, s_val, ax_val] = slice_items[i];
        int start = b_val;
        if (idx_flag) {
            ax_val = ax_val < 0 ? layout.shape[axis] + ax_val : ax_val;
            offset += ax_val * layout.stride[axis] * dtype_size;
        } else {
            start = std::max(start, 0);
            offset += start * layout.stride[axis] * dtype_size;
        }
    }

    // memory forward
    return {Tensor::make(src->blob(), src->offset() + offset, res_layout)};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    return layout_checker;
}

OP_TRAIT_REG(Subtensor, Subtensor, opr::Subtensor)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();

}  // namespace subtensor
}  // namespace

}  // namespace mgb::imperative