#include <numeric>
#include "megbrain/graph/helper.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {
namespace meshgrid {
SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    return SmallVector<VarNode::LayoutConstraintCallback>(inputs.size());
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    for (size_t i = 0; i < inputs.size() - 1; i++) {
        mgb_assert(inputs[i].layout.dtype == inputs[i + 1].layout.dtype);
        mgb_assert(inputs[i].comp_node == inputs[i + 1].comp_node);
    }
    auto&& op = def.cast_final_safe<MeshGrid>();
    mgb_assert(op.indexing == "xy" || op.indexing == "ij");
    bool success = true;
    SmallVector<size_t> shp;
    for (size_t i = 0; i < inputs.size(); i++) {
        mgb_assert(inputs[i].layout.ndim <= 1);
        if (inputs[i].layout.ndim == 0) {
            success = false;
        }
        shp.push_back(inputs[i].layout.total_nr_elems());
    }
    if (op.indexing == "xy" and shp.size() >= 2) {
        std::swap(shp[0], shp[1]);
    }
    TensorShape tshp(shp);
    SmallVector<LogicalTensorDesc> descs;

    for (size_t i = 0; i < inputs.size(); i++) {
        if (success) {
            descs.push_back(
                    {TensorLayout(tshp, inputs[0].layout.dtype), inputs[0].comp_node});
        } else {
            descs.push_back(
                    {TensorLayout(inputs[0].layout.dtype), inputs[0].comp_node});
        }
    }
    return {descs, success};
}
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<MeshGrid>();
    std::vector<size_t> indexs(inputs.size());
    std::iota(indexs.begin(), indexs.end(), 0);
    auto cn = inputs[0]->comp_node();
    auto graph = inputs[0]->owner_graph();
    if (op.indexing == "xy") {
        if (indexs.size() >= 2) {
            std::swap(indexs[0], indexs[1]);
        }
    } else {
        mgb_assert(op.indexing == "ij", "meshgrid only support \"ij\" or \"xy\"");
    }
    VarNodeArray shps;
    for (size_t ind = 0; ind < inputs.size(); ind++) {
        auto&& inp = inputs[indexs[ind]];
        shps.push_back(opr::GetVarShape::make(inp).node());
    }
    VarNode* tshp = opr::Concat::make(shps, 0, cn).node();
    VarNodeArray results;
    auto t_ndim = inputs.size();
    for (size_t ind = 0; ind < inputs.size(); ind++) {
        auto axis = indexs[ind];
        HostTensorND hv = HostTensorND(cn, {t_ndim}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        std::fill_n(ptr, t_ndim, 1);
        ptr[axis] = -1;
        auto shp = opr::ImmutableTensor::make(*graph, hv, cn).node();
        auto tmp = opr::Reshape::make(inputs[ind], shp, axis).node();
        results.push_back(opr::Broadcast::make(tmp, tshp).node());
    }
    return results;
}
SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<MeshGrid>();
    TensorShape tshp;
    TensorShape view_shp;
    tshp.ndim = inputs.size();
    view_shp.ndim = inputs.size();
    std::vector<size_t> indexs(inputs.size());
    std::iota(indexs.begin(), indexs.end(), 0);

    if (op.indexing == "xy") {
        if (indexs.size() >= 2) {
            std::swap(indexs[0], indexs[1]);
        }
    } else {
        mgb_assert(op.indexing == "ij", "meshgrid only support \"ij\" or \"xy\"");
    }
    for (size_t ind = 0; ind < inputs.size(); ind++) {
        auto&& inp = inputs[indexs[ind]];
        mgb_assert(inp->layout().ndim <= 1);
        tshp[ind] = inp->layout().total_nr_elems();
        view_shp[ind] = 1;
    }
    SmallVector<TensorPtr> grids;
    for (size_t i = 0; i < inputs.size(); i++) {
        auto&& src = inputs[i];
        TensorLayout layout;
        view_shp[indexs[i]] = src->layout().total_nr_elems();
        mgb_assert(src->layout().try_reshape(layout, view_shp));
        layout = layout.broadcast(tshp);
        view_shp[indexs[i]] = 1;
        grids.push_back(Tensor::make(src->blob(), src->offset(), layout));
    }
    return grids;
}
OP_TRAIT_REG(MeshGrid, MeshGrid)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace meshgrid
namespace broadcast {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    node_->cast_final_safe<opr::Broadcast>();
    return Broadcast::make();
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 2, "Broadcast expects 2 inputs; got %lu actually", nr_inp);
    OperatorNodeConfig config{op.make_name()};
    return opr::Broadcast::make(inputs[0], inputs[1], config);
}

bool valid_broadcast(const TensorShape& src_shape, const TensorShape& tar_shape) {
    size_t src_ndim = src_shape.ndim, tar_ndim = tar_shape.ndim;
    if (src_ndim > tar_ndim) {
        return false;
    }
    size_t min_ndim = src_ndim;
    for (size_t i = 0; i < min_ndim; ++i) {
        if (src_shape[src_ndim - i - 1] != 1 &&
            src_shape[src_ndim - i - 1] != tar_shape[tar_ndim - i - 1]) {
            return false;
        }
    }
    return true;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    auto&& src = inputs[0];
    TensorShape out_shape;
    if (nr_inp == 1) {
        out_shape.ndim = op.shape.size();
        for (size_t i = 0; i < out_shape.ndim; ++i) {
            out_shape[i] = op.shape[i];
        }
    } else {
        auto&& tshp = inputs[1];
        if (tshp.layout.ndim == 0 || tshp.value.empty()) {
            out_shape.ndim = 0;
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
        mgb_assert(
                tshp.layout.ndim == 1,
                "target shape of Broadcast expects ndim=1; got ndim=%lu actually",
                tshp.layout.ndim);
        size_t target_ndim = tshp.layout.shape[0];
        out_shape.ndim = target_ndim;
        auto* ptr = tshp.value.ptr<dt_int32>();
        for (size_t i = 0; i < target_ndim; ++i) {
            out_shape[i] = ptr[i];
        }
    }
    mgb_assert(
            valid_broadcast(src.layout, out_shape),
            "the input shape %s can not be broadcasted to target shape %s",
            src.layout.to_string().c_str(), out_shape.to_string().c_str());
    return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<Broadcast>();
    size_t nr_inp = inputs.size();
    TensorShape tshp;
    auto&& src = inputs[0];
    auto slayout = src->layout();
    if (nr_inp == 1) {
        tshp.ndim = op.shape.size();
        for (size_t i = 0; i < tshp.ndim; ++i) {
            tshp[i] = op.shape[i];
        }
    } else {
        auto&& tshp_nd = inputs[1];
        cg::copy_tensor_value_to_shape(
                tshp, tshp_nd->get_value().proxy_to_default_cpu());
    }
    if (tshp.is_empty()) {
        return {Tensor::make(TensorLayout(tshp, src->dtype()), src->comp_node())};
    }
    TensorLayout tlayout = slayout.broadcast(tshp);
    // memory forward
    return {Tensor::make(src->blob(), src->offset(), tlayout)};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    return layout_checker;
}

OP_TRAIT_REG(Broadcast, Broadcast, opr::Broadcast)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace broadcast

namespace reshape {

auto make_from_op_node(const cg::OperatorNodeBase* node) {
    auto& opr = node->cast_final_safe<opr::Reshape>();
    return Reshape::make(opr.param(), std::vector<int32_t>());
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Reshape&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Reshape::make(inputs[0], inputs[1], op.param(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op = def.cast_final_safe<Reshape>();
    size_t nr_inp = inputs.size();
    auto&& src = inputs[0];

    TensorShape out_shape;

    if (nr_inp == 1) {
        if (src.layout.ndim == 0 && op.axis != opr::Reshape::Param::INVALID_AXIS) {
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
        out_shape.ndim = op.shape.size();
        for (size_t i = 0; i < out_shape.ndim; ++i) {
            out_shape[i] = op.shape[i];
        }
        if (src.layout.ndim == 0) {
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
    } else {
        auto&& tshp = inputs[1];
        if (tshp.layout.ndim == 0 || tshp.value.empty()) {
            out_shape.ndim = 0;
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
        mgb_assert(
                tshp.layout.ndim == 1,
                "target shape of Reshape expects ndim=1; got ndim=%lu actually",
                tshp.layout.ndim);
        if (src.layout.ndim == 0 && op.axis != opr::Reshape::Param::INVALID_AXIS) {
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
        size_t target_ndim = tshp.layout.shape[0];
        out_shape.ndim = target_ndim;
        auto* ptr = tshp.value.ptr<dt_int32>();
        for (size_t i = 0; i < target_ndim; ++i) {
            out_shape[i] = ptr[i];
        }
        if (src.layout.ndim == 0) {
            return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}},
                    false};
        }
    }
    if (op.axis != opr::Reshape::Param::INVALID_AXIS) {
        out_shape[op.axis] = 1;
        mgb_assert(
                src.layout.total_nr_elems() % out_shape.total_nr_elems() == 0,
                "can not reshape from %s to %s", src.layout.to_string().c_str(),
                out_shape.to_string().c_str());
        out_shape[op.axis] = src.layout.total_nr_elems() / out_shape.total_nr_elems();
    } else {
        mgb_assert(
                src.layout.total_nr_elems() == out_shape.total_nr_elems(),
                "can not reshape from %s to %s", src.layout.to_string().c_str(),
                out_shape.to_string().c_str());
    }
    return {{{TensorLayout(out_shape, src.layout.dtype), src.comp_node}}, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op = def.cast_final_safe<Reshape>();
    size_t nr_inp = inputs.size();
    auto&& src = inputs[0];
    auto slayout = src->layout();
    TensorShape tshp;

    if (nr_inp == 1) {
        tshp.ndim = op.shape.size();
        for (size_t i = 0; i < tshp.ndim; ++i) {
            tshp[i] = op.shape[i];
        }
    } else {
        auto&& tshp_nd = inputs[1];

        cg::copy_tensor_value_to_shape(
                tshp, tshp_nd->get_value().proxy_to_default_cpu());
    }
    if (op.axis != opr::Reshape::Param::INVALID_AXIS) {
        tshp[op.axis] = 1;
        tshp[op.axis] = src->layout().total_nr_elems() / tshp.total_nr_elems();
    }
    TensorLayout tlayout;
    mgb_assert(slayout.try_reshape(tlayout, tshp));
    return {Tensor::make(src->blob(), src->offset(), tlayout)};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    auto&& op = def.cast_final_safe<Reshape>();
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = [&](const TensorLayout& layout) {
        TensorShape tshp;
        TensorLayout ret;
        if (inputs.size() == 1) {
            tshp.ndim = op.shape.size();
            for (size_t i = 0; i < tshp.ndim; ++i) {
                tshp[i] = op.shape[i];
            }
        } else {
            cg::copy_tensor_value_to_shape(
                    tshp, inputs[1]->get_value().proxy_to_default_cpu());
        }
        if (op.axis != opr::Reshape::Param::INVALID_AXIS) {
            tshp[op.axis] = 1;
            tshp[op.axis] = layout.total_nr_elems() / tshp.total_nr_elems();
        }
        if (layout.try_reshape(ret, tshp)) {
            return true;
        } else {
            return false;
        }
    };
    return layout_checker;
}

OP_TRAIT_REG(Reshape, Reshape, opr::Reshape)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .make_from_op_node(make_from_op_node)
        .fallback();
}  // namespace reshape

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
