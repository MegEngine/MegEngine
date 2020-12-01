/**
 * \file src/opr/impl/tensor_manip.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/io.h"
#include "megbrain/graph/event.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/graph/exc_extra_info.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace intl;

/* f{{{ ======================= local utils ======================= */
namespace {
    using OptionalAxis = megdnn::param::OptionalAxisV1;
    //! check whether shp is GetVarShape(a)
    bool check_is_shape_of(SymbolVar shp, SymbolVar a) {
#if MGB_BUILD_SLIM_SERVING
        return false;
#else
        auto op = shp.node()->owner_opr();
        if (op->same_type<GetVarShape>() && op->input().size() == 1 &&
            op->input()[0] == a.node() &&
            op->cast_final<GetVarShape>().param().axis ==
                    OptionalAxis::INVALID_AXIS) {
            return true;
        }
        using namespace cg::static_infer;
        auto &&mgr = a.node()->owner_graph()->static_infer_manager();
        if ((mgr.get_infer_type(shp.node()).value & InferType::CONST) &&
                (mgr.get_infer_type(a.node()).shape & InferType::CONST)) {
            auto &&a_shp = mgr.infer_shape(a.node());
            auto &&shp_val = mgr.infer_value(shp.node());
            TensorShape shp_shp;
            cg::copy_tensor_value_to_shape(shp_shp, shp_val);
            return a_shp.eq_shape(shp_shp);
        }
        return false;
#endif
    }

#if !MGB_BUILD_SLIM_SERVING
    // return x such that shape_of(var) == x
    GetVarShape* get_shape_shortcut(VarNode *var) {
        auto opr = var->owner_opr();
        auto otype = opr->dyn_typeinfo();
        if (!(otype == Reshape::typeinfo() &&
              opr->cast_final<Reshape>().param().axis ==
                      OptionalAxis::INVALID_AXIS) &&
            otype != Broadcast::typeinfo()) {
            return nullptr;
        }
        auto i1 = opr->input(1)->owner_opr();
        if (i1->same_type<GetVarShape>())
            return &i1->cast_final<GetVarShape>();
        return nullptr;
    }
#endif
} // anonymous namespace
// f}}}

/* f{{{ ======================= GetVarShape ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GetVarShape);
GetVarShape::GetVarShape(const VarNodeArrayView &inp, Param axis,
        const OperatorNodeConfig &config):
    Super(inp.at(0)->owner_graph(), config, "shape_of", inp),
    m_axis{axis}
{
    m_src_shapes.resize(inp.size());
    for (auto i: inp)
        add_input({i});
    add_input({}, AddInputSortType::ALL);
    add_output(None)->dtype(dtype::Int32());
    add_equivalence_component<PODHash<Param>>(&m_axis);
    mgb_assert(abs(m_axis.axis) <= m_axis.MAX_NDIM);
}

void GetVarShape::update_cached_shape() {
    TensorShape ishp;
    if (m_src_shapes.size() == 1) {
        ishp = m_src_shapes[0];
    } else {
        megdnn::Elemwise::deduce_shape(m_src_shapes, ishp);
    }
    mgb_assert(ishp.ndim);
    // check whether m_cached_shape is valid and update it if not
    if (m_axis.axis != OptionalAxis::INVALID_AXIS) {
        int axis = m_axis.axis;
        if (axis < 0) {
            axis += ishp.ndim;
        }
        mgb_assert(axis >= 0 && axis < (int)ishp.ndim);
        if (m_cached_shape.ndim == 1 &&
            m_cached_shape.shape[0] == ishp.shape[axis])
            return;
        m_cached_shape = {ishp.shape[axis]};
    } else {
        if (m_cached_shape.eq_shape(ishp))
            return;
        m_cached_shape = ishp;
    }

    cg::copy_shape_to_tensor_value(m_cached_shape_cpu_v, m_cached_shape);
    m_cached_shape_dev_v_synced = false;
}

void GetVarShape::scn_do_execute() {
    for (size_t i = 0; i < m_src_shapes.size(); ++ i) {
        m_src_shapes[i] = input()[i]->shape();
    }
    update_cached_shape();
    if (!m_cached_shape_dev_v_synced) {
        m_cached_shape_dev_v.copy_from(m_cached_shape_cpu_v);
        m_cached_shape_dev_v_synced = true;
    }
    output(0)->dev_tensor().copy_from_fixlayout(m_cached_shape_dev_v);
}

void GetVarShape::update_for_static_infer(const cg::static_infer::InpVal &inp) {
    for (size_t i = 0; i < m_src_shapes.size(); ++ i) {
        m_src_shapes[i] = inp.val.at(i).shape();
    }
    update_cached_shape();
}

void GetVarShape::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_shape = [this](TensorShape &dest, const InpVal &inp) {
        update_for_static_infer(inp);
        dest = m_cached_shape_cpu_v.shape();
        return true;
    };

    auto infer_value = [this](DeviceTensorND &dest, const InpVal &inp) {
        update_for_static_infer(inp);
        dest = m_cached_shape_cpu_v;
        return true;
    };

    DepVal deps;
    for (auto i: input()) {
        deps.push_back({i, DepType::SHAPE});
    }

    auto &&mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0),
            {SourceType::DEP, deps, infer_shape});
    mgr.register_value_infer(output(0),
            {SourceType::DEP, deps, infer_value});
}
#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(GetVarShape) {
    MGB_MARK_USED_VAR(wrt_idx);
    MGB_MARK_USED_VAR(out_grad);
    return nullptr;
}
#endif

SymbolVar GetVarShape::make(const VarNodeArrayView& inp, Param param,
                            const OperatorNodeConfig& config) {
    mgb_assert(!inp.empty());

#if !MGB_BUILD_SLIM_SERVING
    // try to apply shortcut and omit scalar shapes to optimize
    VarNodeArray inp_vp;
    inp_vp.reserve(inp.size());
    auto&& mgr = inp[0]->owner_graph()->static_infer_manager();
    for (auto var : inp) {
        auto&& it = mgr.get_infer_type(var);
        if (it.shape & cg::static_infer::InferType::CONST) {
            if (mgr.infer_shape(var).is_scalar()) {
                // scalar does not affect broadcast result
                continue;
            }
        }
        if (auto opr = get_shape_shortcut(var)) {
            // current var replaced by a shortcut
            auto&& op_inp = opr->input();
            inp_vp.insert(inp_vp.end(), op_inp.begin(), op_inp.end());
            continue;
        }
        inp_vp.push_back(var);
    }
    if (inp_vp.empty()) {
        // all inputs are scalar
        mgb_assert(param.axis == OptionalAxis::INVALID_AXIS || param.axis == 0);
        return SymbolVar{inp[0]}.make_scalar(1);
    }
#else
    auto&& inp_vp = inp;
#endif
    return SymbolVar{inp[0]}.insert_single_output_opr<GetVarShape>(
            inp_vp, param, config);
}

cg::OperatorNodeBase::NodeProp* GetVarShape::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using DT = NodeProp::DepType;
    SmallVector<DT> dt(input().size(), DT::SHAPE);
    prop->reset_dep_type(input(), dt);
    return prop;
}

class GetVarShape::ShapeDevValueExecDep final : public ExecDependency {
    DeviceTensorStorage m_val;

public:
    explicit ShapeDevValueExecDep(DeviceTensorStorage val)
            : m_val(std::move(val)) {}
};

void GetVarShape::record_execute_deps(ExecDependencyArray& deps) {
    deps.emplace_back(std::make_unique<ShapeDevValueExecDep>(
            m_cached_shape_dev_v.storage()));
}

// f}}}

/* f{{{ ======================= ReshapeBrdcastHelper ======================= */

void ReshapeBrdcastHelper::reshapebrdcast_init(VarNode *inp, VarNode *tshp) {
    add_input({inp, tshp});
    add_output(None)->dtype(inp->dtype())
                    .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    if (reshapebrdcast_output_shape_need_input_shape())
        outshape_by_symvar_enable(1, 1);
    else
        outshape_by_symvar_enable(0, 1);
}


void ReshapeBrdcastHelper::mem_plan_fwd_in2out_readonly() {
    auto &&tshape = output(0)->shape();

    auto inp_layout = input(0)->layout();
    auto dst_layout = reshapebrdcast_get_dest_layout(inp_layout, tshape);
    if (!dst_layout.valid()) {
        // retry after making input contiguous
        mgb_assert(dyn_typeinfo() == Reshape::typeinfo());
        inp_layout.init_contiguous_stride(input(0)->shape());
        dst_layout = reshapebrdcast_get_dest_layout(inp_layout, tshape);
        mgb_assert(dst_layout.valid());
        m_rofwd_subspec = SubTensorSpec::make_from_layout(dst_layout.val());
        m_incompatible_inp_layout = true;
        return;
    }
    m_rofwd_subspec = SubTensorSpec::make_from_layout(dst_layout.val());
    m_incompatible_inp_layout = false;
    rofwd_init_mem_plan();
}

void ReshapeBrdcastHelper::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest,
        const ShapeInferInfo &shpinfo) {
    if (reshapebrdcast_output_shape_need_input_shape()) {
        TensorShape oshp_given;
        cg::copy_tensor_value_to_shape(oshp_given,
                                       *shpinfo.shpval_inp_val.at(0));
        TensorLayout src;
        src.init_contiguous_stride(shpinfo.shape_inp_shp.at(0));
        dest = reshapebrdcast_get_dest_layout(src, oshp_given).val();
    } else {
        cg::copy_tensor_value_to_shape(dest, *shpinfo.shpval_inp_val.at(0));
    }
}

void ReshapeBrdcastHelper::scn_do_execute() {
    if (m_incompatible_inp_layout) {
        // only happens in reshape
        auto &&iv = input(0)->dev_tensor();
        auto ishp = iv.shape();
        auto &&ov = output(0)->dev_tensor();
        mgb_assert(ishp.total_nr_elems() == ov.shape().total_nr_elems());
        ov.sub(SubTensorSpec::make_from_layout({ishp, iv.dtype()})).
            copy_from_fixlayout(iv);
    } else
        rofwd_execute();
}

void ReshapeBrdcastHelper::add_input_layout_constraint() {
    if (!cg::is_static_var_value(input(1)))
        return;

    auto check_layout = [this](const TensorLayout &layout) {
        MGB_TRY {
            TensorShape oshp;
            outshape_by_symvar_do_get_output_shape(
                    oshp, outshape_by_symvar_get_shape_infer_info());
            return reshapebrdcast_get_dest_layout(layout, oshp).valid();
        } MGB_CATCH(MegBrainError &exc,  {
            if (!exc.extra_info())
                cg::OperatorNodeExcExtraInfo::record(this, exc);
            throw;
        })
    };
    input(0)->add_layout_constraint(check_layout);
}

void ReshapeBrdcastHelper::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    using namespace cg::static_infer;
    auto infer_value = [this](DeviceTensorND &dest, const InpVal &inp) {
        TensorShape oshp;
        cg::copy_tensor_value_to_shape(oshp, inp.val.at(1).value());
        auto &&iv = inp.val[0].value();
        auto sub_layout = reshapebrdcast_get_dest_layout(iv.layout(), oshp);
        if (sub_layout.valid()) {
            dest = const_cast<DeviceTensorND&>(iv).sub(
                    SubTensorSpec::make_from_layout(sub_layout.val()));
        } else {
            // use contig dest
            dest = {};
            dest.copy_from(iv);
            sub_layout = reshapebrdcast_get_dest_layout(dest.layout(), oshp);
            mgb_assert(sub_layout.valid());
            dest = dest.sub(SubTensorSpec::make_from_layout(sub_layout.val()));
        }
        return true;
    };

    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP,
            {{input(0), DepType::VALUE}, {input(1), DepType::VALUE}},
            infer_value});
}

ReshapeBrdcastHelper::NodeProp*
ReshapeBrdcastHelper::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0),
                                   NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

// f}}}

/* f{{{ ======================= Reshape ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Reshape);

Reshape::Reshape(VarNode *inp, VarNode *tshp, Param unspec_axis,
                const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "reshape", {inp}},
    m_unspec_axis{unspec_axis}
{
    reshapebrdcast_init(inp, tshp);
    add_equivalence_component<PODHash<Param>>(&m_unspec_axis);
}

SymbolVar Reshape::make(SymbolVar inp, SymbolVar tshp,
        Param unspec_axis, const OperatorNodeConfig &config) {
    if (check_is_shape_of(tshp, inp))
        return inp;
    return inp.insert_single_output_opr<Reshape>(
            inp.node(), tshp.node(), unspec_axis, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Reshape) {
    if (wrt_idx)
        return InvalidGrad::make(opr, wrt_idx);
    return Reshape::make(out_grad[0], GetVarShape::make(opr.input(0))).node();
}
#endif

Maybe<TensorLayout> Reshape::reshapebrdcast_get_dest_layout(
        const TensorLayout &src, const TensorShape &tshape) const {
    if (m_unspec_axis.axis == OptionalAxis::INVALID_AXIS) {
        TensorLayout ret;
        if (src.try_reshape(ret, tshape))
            return ret;
        return None;
    }

    int original_unspec = m_unspec_axis.axis;
    if (original_unspec < 0) {
        original_unspec += tshape.ndim;
    }
    size_t unspec = original_unspec;
    mgb_assert(unspec < tshape.ndim);
    auto actual_tshape = tshape;
    size_t rem_nr_elem = 1;
    for (size_t i = 0; i < tshape.ndim; ++ i) {
        if (i != unspec)
            rem_nr_elem *= tshape.shape[i];
    }
    auto tot_nr_elem = src.total_nr_elems();
    actual_tshape.shape[unspec] = 0;
    mgb_throw_if(!rem_nr_elem || tot_nr_elem % rem_nr_elem, TensorReshapeError,
            "could not reshape: src=%s tshape=%s unspec_axis=%zd",
            static_cast<const TensorShape&>(src).to_string().c_str(),
            actual_tshape.to_string().c_str(),
            unspec);
    actual_tshape.shape[unspec] = tot_nr_elem / rem_nr_elem;
    TensorLayout ret;
    if (src.try_reshape(ret, actual_tshape))
        return ret;
    return None;
}

bool Reshape::reshapebrdcast_output_shape_need_input_shape() const {
    return m_unspec_axis.axis != OptionalAxis::INVALID_AXIS;
}

// f}}}

/* f{{{ ======================= Broadcast ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Broadcast);

Broadcast::Broadcast(VarNode *inp, VarNode *tshp,
        const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "broadcast", {inp}}
{
    reshapebrdcast_init(inp, tshp);
}


SymbolVar Broadcast::make(SymbolVar inp, SymbolVar tshp,
        const OperatorNodeConfig &config) {
    if (check_is_shape_of(tshp, inp))
        return inp;
    return inp.insert_single_output_opr<Broadcast>(
            inp.node(), tshp.node(), config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Broadcast) {
    if (wrt_idx)
        return InvalidGrad::make(opr, wrt_idx);
    return Reduce::make(out_grad.at(0), Reduce::Mode::SUM,
            GetVarShape::make(opr.input(0))).node();
}
#endif

Maybe<TensorLayout> Broadcast::reshapebrdcast_get_dest_layout(
        const TensorLayout &src, const TensorShape &tshape) const {
    return src.broadcast(tshape);
}

bool Broadcast::reshapebrdcast_output_shape_need_input_shape() const {
    return false;
}

// f}}}

/* f{{{ ======================= AxisManipOprBase ======================= */
void AxisManipOprBase::mem_plan_fwd_in2out_readonly() {
    m_rofwd_subspec = SubTensorSpec::make_from_layout(
            axis_manip_get_output_layout(input(0)->layout()));
    rofwd_init_mem_plan();
}

void AxisManipOprBase::scn_do_execute() {
    rofwd_execute();
}

void AxisManipOprBase::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_shape = [this](TensorShape &dest, const InpVal &inp) {
        dest = axis_manip_get_output_layout({
                inp.val.at(0).shape(), input(0)->dtype()});
        return true;
    };
    auto infer_value = [this](DeviceTensorND &dest, const InpVal &inp) {
        auto &&iv = inp.val.at(0).value();
        auto oly = axis_manip_get_output_layout(iv.layout());
        dest = const_cast<DeviceTensorND&>(iv).sub(
                SubTensorSpec::make_from_layout(oly));
        return true;
    };
    mgr.register_shape_infer(output(0),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
    mgr.register_value_infer(output(0),
            {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_value});
}

AxisManipOprBase::NodeProp* AxisManipOprBase::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0),
                                   NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

void AxisManipOprBase::axis_manip_init(VarNode* inp) {
    add_input({inp});
    add_output(None)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

// f}}}

/* f{{{ ======================= Dimshuffle ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Dimshuffle);

Dimshuffle::Dimshuffle(VarNode *inp, const std::vector<int> &pattern,
        size_t ndim, const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "dimshuffle", {inp}},
    m_pattern(pattern),
    m_inp_ndim(ndim)
{
    mgb_throw_if(m_pattern.size() > TensorShape::MAX_NDIM,
            GraphError, "Dimshuffle pattern exceeds max length of %zd",
            TensorShape::MAX_NDIM);
    for (auto i: m_pattern) {
        mgb_throw_if(i < -1 || i >= int(ndim), GraphError,
                "bad Dimshuffle pattern");
    }
    axis_manip_init(inp);
    add_equivalence_component<PODHash<int>>(m_pattern.data(), m_pattern.size());
}

SymbolVar Dimshuffle::make(
        SymbolVar inp, const std::vector<int> &pattern,
        size_t ndim, const OperatorNodeConfig &config) {
    if (!ndim)
        ndim = *std::max_element(pattern.begin(), pattern.end()) + 1;
    return inp.insert_single_output_opr<Dimshuffle>(inp.node(),
            pattern, ndim, config);
}

TensorLayout Dimshuffle::axis_manip_get_output_layout(
        const TensorLayout &ily) const {

    mgb_assert(ily.ndim == m_inp_ndim,
            "input ndim mismatch for Dimshuffle: expect=%zd actual=%zd",
            m_inp_ndim, ily.ndim);
    TensorLayout oly{ily.dtype};
    oly.ndim = m_pattern.size();

    size_t idx = 0;
    bool input_used[TensorLayout::MAX_NDIM] = {0};
    for (auto i: m_pattern) {
        if (i < 0) {
            oly.shape[idx] = 1;
            oly.stride[idx] = 1;
        } else {
            input_used[i] = true;
            oly.shape[idx] = ily.shape[i];
            oly.stride[idx] = ily.stride[i];
        }
        ++ idx;
    }

    for (size_t i = 0; i < m_inp_ndim; ++ i) {
        mgb_assert(input_used[i] || ily.shape[i] == 1,
                "non-1 dim discarded in Dimshuffle: ishp=%s dim=%zd",
                static_cast<const TensorShape&>(ily).to_string().c_str(),
                i);
    }
    return oly;
}

VarNode* Dimshuffle::grad(
        size_t /*wrt_idx*/, const VarNodeArray &out_grad) const {

    std::vector<int> back(m_inp_ndim, -1);
    for (size_t i = 0; i < m_pattern.size(); i ++) {
        // outdim[i] is indim[j]
        auto j = m_pattern[i];
        if (j >= 0) {
            mgb_assert(back[j] == -1,
                    "taking grad for Dimshuffle with duplicated "
                    "input axis unsupported");
            back[j] = i;
        }
    }
    return Dimshuffle::make(out_grad.at(0), back, m_pattern.size()).node();
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Dimshuffle) {
    return opr.grad(wrt_idx, out_grad);
}
#endif

// f}}}

/* f{{{ ======================= AxisAddRemove ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AxisAddRemove);

AxisAddRemove::AxisAddRemove(
        VarNode *inp, const std::vector<AxisDesc> &desc,
        const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "axis_add_rm", {inp}},
    m_desc(desc)
{
    mgb_throw_if(desc.empty(), GraphError,
            "desc for AxisAddRemove could not be empty");
    axis_manip_init(inp);
    add_equivalence_component<PODHash<AxisDesc>>(m_desc.data(), m_desc.size());
}

SymbolVar AxisAddRemove::make(SymbolVar inp,
        const std::vector<AxisDesc> &desc,
        const OperatorNodeConfig &config) {
    return inp.insert_single_output_opr<AxisAddRemove>(inp.node(), desc, config);
}

TensorLayout AxisAddRemove::axis_manip_get_output_layout(
        const TensorLayout &input_layout) const {
    auto layout = input_layout;

    for (auto &&i: m_desc) {
        using M = AxisDesc::Method;
        switch (i.method) {
            case M::REMOVE:
            {
                auto axis = i.axis.get(layout.ndim);
                if (layout.ndim == 1) {
                    mgb_assert(layout.shape[0] == 1 && axis == 0,
                            "can not remove axis %zu from tensor of shape=%s",
                            axis,
                            layout.megdnn::TensorShape::to_string().c_str());
                } else {
                    mgb_assert(axis < layout.ndim &&
                            layout.shape[axis] == 1,
                            "can not remove axis %zu from tensor of shape=%s",
                            axis,
                            layout.megdnn::TensorShape::to_string().c_str());
                    layout.remove_axis_inplace(axis);
                }
                break;
            }
            case M::ADD_1:
                layout.add_axis_cont_inplace(i.axis.get(layout.ndim + 1));
                break;
        }
    }
    return layout;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(AxisAddRemove) {
    MGB_MARK_USED_VAR(wrt_idx);
    return Reshape::make(out_grad[0], GetVarShape::make(opr.input(0))).node();
}
#endif

// f}}}

/* f{{{ ======================= Subtensor ======================= */

MGB_IMPL_FANCY_INDEXING_OPR_GET(Subtensor, "subtensor", true);

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Subtensor) {
    if (wrt_idx)
        return InvalidGrad::make(opr, wrt_idx);

    return IncrSubtensor::make(
            SymbolVar{opr.input(0)}.fill_retain_dtype(0),
            out_grad.at(0), opr.index_desc()).node();
}
#endif

void Subtensor::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    DepVal deps;

    // shape inference only needs slices
    deps.push_back({input(0), DepType::SHAPE});
    for (size_t i = 1; i < m_input2idxonly_axis_indexer.size(); ++ i) {
        if (!m_input2idxonly_axis_indexer[i])
            deps.push_back({input(i), DepType::VALUE});
    }
    auto infer_shape = [this](TensorShape &dest, const InpVal &inp) {
        auto &&ishp = inp.val[0].shape();
        auto subspec = fancy_indexing_make_sub_spec(
                {ishp, input(0)->dtype()}, inp, 1, true);
        dest = subspec.layout();
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), {SourceType::DEP, deps, infer_shape});

    deps.clear();
    for (auto i: input())
        deps.push_back({i, DepType::VALUE});
    deps[0].type = DepType::VALUE;
    auto infer_value = [this](DeviceTensorND &dest, const InpVal &inp) {
        auto &&iv = inp.val[0].value();
        auto subspec = fancy_indexing_make_sub_spec(iv.layout(), inp, 1);
        dest = const_cast<DeviceTensorND&>(iv).sub(subspec);
        return true;
    };
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), {SourceType::DEP, deps, infer_value});
}

void Subtensor::scn_do_execute() {
    rofwd_execute();
}

void Subtensor::mem_plan_fwd_in2out_readonly() {
    m_rofwd_subspec = fancy_indexing_make_sub_spec(input(0)->layout());
    rofwd_init_mem_plan();
}

void Subtensor::init_rt_force_dynamic_mem_alloc_imply_chain() {
    auto inp = input(0), out = output(0);
    inp->add_rt_force_dynamic_mem_alloc_imply_chain(out);
    out->add_rt_force_dynamic_mem_alloc_imply_chain(inp);
}

// f}}}

/* f{{{ ================== ModifySubtensorImplHelper ================== */

void ModifySubtensorImplHelper::scn_do_execute() {
    auto mod = fancy_indexing_get_tensors_for_modify_in_scn_do_execute();
    modify(mod.first, mod.second);
}

void ModifySubtensorImplHelper::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();

    // try to register shape infer with subtensor shape check
    auto try_infer_shape_with_check = [&]() -> bool{

        if (!cg::is_static_var_shape(input(0)) ||
                !cg::is_static_var_shape(input(1)))
            return false;
        for (size_t i = 2; i < input().size(); ++ i) {
            if (!cg::is_static_var_value(input(i)) ||
                !mgr.infer_value_fallible(input(i)))
                return false;
        }

        auto infer_shape = [this](TensorShape &dest, const InpVal &inp) {
            dest = inp.val.at(0).shape();
            // throw exception if shapes mismatch
            auto subspec = fancy_indexing_make_sub_spec(
                    {dest, input(0)->dtype()}, inp, 2);
            auto &&subshp = inp.val.at(1).shape();
            mgb_throw_if(!subspec.layout().eq_shape(subshp), TensorReshapeError,
                    "SetSubtensor shape mismatch: subspec=%s value_shape=%s",
                    subspec.layout().TensorShape::to_string().c_str(),
                    subshp.to_string().c_str());
            return true;
        };
        DepVal deps;
        for (auto i: input())
            deps.push_back({i, DepType::VALUE});
        deps[0].type = deps[1].type = DepType::SHAPE;
        mgr.register_shape_infer(output(0), {
                SourceType::DEP, deps, infer_shape});
        return true;
    };

    if (has_input_tensor_replacer()) {
        mgr.register_shape_infer(output(0), ShapeInferDesc::make_const({}));
    } else {
        if (!try_infer_shape_with_check()) {
            auto infer_shape = [](TensorShape &dest, const InpVal &inp) {
                dest = inp.val.at(0).shape();
                return true;
            };
            mgr.register_shape_infer(output(0), {
                    SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
        }
    }

    auto infer_value = [this](DeviceTensorND &dest, const InpVal &inp) {
        dest.copy_from(inp.val.at(0).value());
        auto subspec = fancy_indexing_make_sub_spec(dest.layout(), inp, 2);
        auto dsub = dest.sub(subspec);
        modify(dsub, inp.val.at(1).value());
        return true;
    };
    DepVal value_deps;
    for (auto i: input())
        value_deps.push_back({i, DepType::VALUE});

    mgr.register_value_infer(output(0), {
            SourceType::DEP, value_deps, infer_value});
}

// f}}}

/* f{{{ ======================= SetSubtensor ======================= */

MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(SetSubtensor, "set_subtensor", true);

void SetSubtensor::modify(DeviceTensorND &sub, const DeviceTensorND &val) {
    sub.copy_from_fixlayout(val);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SetSubtensor) {
    if (wrt_idx >= 2)
        return InvalidGrad::make(opr, wrt_idx);
    if (wrt_idx == 0) {
        return SetSubtensor::make(out_grad.at(0),
                SymbolVar{opr.input(1)}.fill_retain_dtype(0),
                opr.index_desc()).node();
    }
    return Subtensor::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

// f}}}

/* f{{{ ======================= IncrSubtensor ======================= */

MGB_IMPL_FANCY_INDEXING_OPR_MODIFY(IncrSubtensor, "incr_subtensor", true);

void IncrSubtensor::modify(DeviceTensorND &sub, const DeviceTensorND &val) {
    CompNode opr_comp_node;
    if (sub.comp_node().locator().device ==
            CompNode::Locator::DEVICE_CPU_DEFAULT) {
        // for static infer
        opr_comp_node = CompNode::default_cpu();
    } else {
        opr_comp_node = comp_node();
    }
    auto opr = intl::get_megdnn_global_opr<megdnn::AddUpdate>(opr_comp_node);
    opr->exec(sub.as_megdnn(), val.as_megdnn());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(IncrSubtensor) {
    if (wrt_idx >= 2)
        return InvalidGrad::make(opr, wrt_idx);
    if (wrt_idx == 0) {
        return out_grad.at(0);
    }
    return Subtensor::make(out_grad.at(0), opr.index_desc()).node();
}
#endif

// f}}}

/* f{{{ ======================= IndexAt ======================= */
SymbolVar IndexAt::make(SymbolVar inp,
        const std::vector<std::pair<size_t, SymbolVar>> &index,
        const OperatorNodeConfig &config) {
    Subtensor::IndexDesc desc;
    for (auto &&i: index) {
        desc.emplace_back();
        desc.back().axis = i.first;
        desc.back().idx = i.second;
    }
    return Subtensor::make(inp, desc, config);
}

// f}}}

/* f{{{ ======================= Split ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Split);

Split::Options Split::Options::make_average(int axis, size_t nr_part) {
    auto cb = [nr_part](size_t size) {
        std::vector<size_t> part(nr_part, size / nr_part);
        for (size_t i = 0, it = size % nr_part; i < it; ++ i)
            ++ part[i];
        return part;
    };
    return make_callback(axis, nr_part, cb);
}

Split::Options Split::Options::make_partition(int axis,
        const SymbolVarArray &partition) {
    mgb_assert(!partition.empty());
    Options rst;
    rst.method = Method::SPECIFY;
    rst.axis = axis;
    rst.partition = partition;
    return rst;
}

Split::Options Split::Options::make_partition(SymbolVar inp, int axis,
        const std::vector<size_t> &partition) {
    SymbolVarArray sym_partition;
    for (auto i: partition)
        sym_partition.push_back(inp.make_scalar(static_cast<int>(i)));
    return make_partition(axis, sym_partition);
}

Split::Options Split::Options::make_callback(
        int axis, size_t nr_part, callback_t callback) {
    mgb_assert(nr_part);
    Options rst;
    rst.method = Method::CALLBACK;
    rst.axis = axis;
    rst.callback = callback;
    rst.nr_part = nr_part;
    return rst;
}

SymbolVarArray Split::make(SymbolVar inp, Options opt,
        const OperatorNodeConfig &config) {
    SymbolVarArray ret;
    auto &&output = inp.node()->owner_graph()->insert_opr(
            std::make_unique<Split>(inp.node(), opt, config))->output();
    for (auto i: output) {
        ret.emplace_back(i);
    }
    return ret;
}

Split::Split(VarNode *inp, const Options &opt, const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "split", {inp}},
    m_opt(opt)
{
    add_input({inp});

    add_equivalence_component<ScalarHash<size_t>>(m_opt.axis);
    if (m_opt.method == Options::Method::SPECIFY) {
        mgb_assert(!m_opt.partition.empty());
        for (auto &&i: m_opt.partition)
            add_input({i.node()});
        outshape_by_symvar_enable(0, 1);
        m_opt.nr_part = m_opt.partition.size();
    }  else {
        // disable dedup
        add_equivalence_component<ScalarHash<void*>>(this);

        mgb_assert(m_opt.method == Options::Method::CALLBACK);
        mgb_assert(m_opt.nr_part);
    }

    for (size_t i = 0; i < m_opt.nr_part; ++ i)
        add_output(ssprintf("o%zd", i))->dtype(inp->dtype());

    m_output_spec.resize(m_opt.nr_part);
}

void Split::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    using namespace std::placeholders;
    auto &&mgr = owner_graph()->static_infer_manager();

    DepVal shp_deps{{input(0), DepType::SHAPE}};
    if (m_opt.method == Options::Method::SPECIFY) {
        for (size_t i = 1; i < input().size(); ++ i)
            shp_deps.push_back({input(i), DepType::VALUE});
    }

    auto infer_value = [this](size_t oidx,
            DeviceTensorND &dest, const InpVal &inp) {
        auto &&cur_shp = m_output_spec[oidx].shape;
        mgb_assert(cur_shp.eq_shape(inp.val[1].shape()));
        auto axis = m_opt.axis;
        if (axis < 0)
            axis += m_output_spec[0].shape.ndim;
        size_t offset = 0;
        for (size_t i = 0; i < oidx; ++ i)
            offset += m_output_spec[i].shape[axis];
        auto &&iv = inp.val[0].value();
        auto subspec = Slice(offset, offset + cur_shp[axis]).apply(
                iv.layout(), axis);
        dest.copy_from(const_cast<DeviceTensorND&>(iv).sub(subspec));
        return true;
    };

    for (size_t i = 0; i < output().size(); ++ i) {
        auto ov = output(i);

        mgr.register_shape_infer(ov,
                {SourceType::DEP, shp_deps, std::bind(
                        &Split::infer_shape, this, i, _1, _2)});

        mgr.register_value_infer(ov, {
                SourceType::DEP,
                {{input(0), DepType::VALUE}, {ov, DepType::SHAPE}},
                std::bind(infer_value, i, _1, _2)});
    }
}

bool Split::infer_shape(size_t out_idx, TensorShape &dest,
        const cg::static_infer::InpVal &inp) {
    if (inp.run_id != m_output_shape_version) {
        std::vector<size_t> partition;
        auto ishp = inp.val.at(0).shape();
        auto axis = m_opt.axis;
        if (axis < 0)
            axis += ishp.ndim;
        if (m_opt.method == Options::Method::SPECIFY) {
            for (size_t i = 0; i < m_opt.nr_part; ++ i) {
                auto &&val = inp.val.at(i + 1).value();
                mgb_assert(val.shape().is_scalar(),
                        "shapes for Split must be scalars");
                size_t cvt;
                static_cast_dtype_safe(&cvt, val.dtype(), val.raw_ptr());
                partition.push_back(cvt);
            }
        } else {
            partition = m_opt.callback(ishp.shape[axis]);
            mgb_assert(partition.size() == m_opt.nr_part,
                    "nr_part=%zu but split callback returned %zu parts",
                    m_opt.nr_part, partition.size());
        }
        size_t size = 0;
        for (size_t i = 0; i < m_opt.nr_part; ++ i) {
            auto p = partition[i];
            mgb_assert(p,
                    "got zero partition size at part %zu, tot_size=%zu",
                    i, ishp.shape[axis]);

            size += p;

            auto &&cur = m_output_spec[i].shape;
            cur = ishp;
            cur.shape[axis] = p;

        }
        mgb_assert(size == ishp.shape[axis],
            "split size sums to %zd, but shape at the axis is %zd",
            size, ishp.shape[axis]);
        m_output_shape_version = inp.run_id;
    }

    dest = m_output_spec.at(out_idx).shape;
    return true;
}

void Split::init_output_comp_node() {
    auto &&conf_node = config().comp_node();
    auto &&cn_opt = owner_graph()->seq_comp_node_optimizer();

    // details of each comp_node specified
    if (conf_node.size() > 1) {
        mgb_assert(conf_node.size() == output().size(),
                "number of CompNodes specified in config should equal to number"
                " of output, but got %zd configured CompNodes while there are"
                " %zd output (node_name=%s node_type=%s)",
                conf_node.size(), output().size(),
                cname(), dyn_typeinfo()->name);
        auto cn0 = input(0)->comp_node();
        for (size_t i = 0; i < output().size(); i ++) {
            auto dvar = output(i);
            dvar->comp_node(conf_node[i]);
            if (conf_node[i].mem_node() != cn0.mem_node())
                cn_opt.register_stream_var(
                        dvar, {CompNode::Stream::COPY,
                               cg::SeqCompNodeOptimizer::StreamPropType::WEAK});
        }
        return;
    }

    CompNode cn;
    if (conf_node.size() == 1) {
        cn = conf_node[0];
    } else {
        cn = input(0)->comp_node();
    }
    for (auto i: output())
        i->comp_node(cn);

    if (cn.mem_node() != input(0)->comp_node().mem_node()) {
        for (auto i: output())
            cn_opt.register_stream_var(
                    i, {CompNode::Stream::COPY,
                        cg::SeqCompNodeOptimizer::StreamPropType::WEAK});
    }
}

cg::OperatorNodeBase::NodeProp* Split::do_make_node_prop() const {
    auto rst = OperatorNodeBase::do_make_node_prop();
    rst->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    outshape_by_symvar_reset_node_dep_type(rst);
    return rst;
}

void Split::do_execute(ExecEnv &env) {
    for (size_t idx = 0; idx < output().size(); ++ idx) {
        auto out = output(idx);

        if (!owner_graph()->var_receiver_in_current_comp_seq(out
                    ).value_needed())
            continue;

        auto runner = [idx, this]() {
            auto &&in = input(0)->dev_tensor();
            auto &&out = output(idx)->dev_tensor();
            auto &&spec = m_output_spec.at(idx);
            owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(
                    this, out.comp_node());
            if (spec.mem_fwd_success) {
                mgb_assert(out.raw_ptr() ==
                        in.raw_ptr() + spec.subspec.offset_byte());
            } else {
                out.comp_node().activate();
                out.copy_from_fixlayout(in.sub(spec.subspec));
            }
            owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
                    this, out.comp_node());
        };
        env.dispatch_on_comp_node(out->comp_node(), runner);
    }
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Split) {
    if (wrt_idx)
        return InvalidGrad::make(opr, wrt_idx);
    mgb_assert(out_grad.size() == opr.output().size());
    SymbolVarArray grad;
    for (size_t i = 0; i < out_grad.size(); ++ i) {
        auto gval = out_grad[i];
        if (!gval) {
            gval = SymbolVar{opr.output(i)}.fill_retain_dtype(0).node();
        }
        grad.emplace_back(gval);
    }
    return Concat::make(grad, opr.options().axis,
            OperatorNodeConfig{}.follow_comp_node(opr.input(0))).node();
}
#endif

void Split::mem_plan_fwd_in2out_readonly() {
    m_readonly_fwd_called = true;
    init_subspec(true);
}

void Split::init_subspec(bool memfwd) {
    auto in = input(0);
    size_t begin = 0, end = 0;
    for (size_t i = 0; i < output().size(); ++ i) {
        auto &&spec = m_output_spec[i];
        auto out = output(i);
        auto real_axis = m_opt.axis;
        if (real_axis < 0)
            real_axis += spec.shape.ndim;
        begin = end;
        mgb_assert(out->shape().eq_shape(spec.shape));
        end = begin + spec.shape.shape[real_axis];
        spec.subspec = Slice(begin, end).apply(in->layout(), real_axis);
        if (out->comp_node() == in->comp_node() && memfwd) {
            spec.mem_fwd_success = out->set_fwd_in2out_readonly(
                    in, spec.subspec);
        } else {
            spec.mem_fwd_success = false;
        }
    }
}

void Split::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {
    // shape infer handled in this class
    MGB_MARK_USED_VAR(dest);
    MGB_MARK_USED_VAR(shpinfo);
    mgb_assert(0);
}

void Split::add_input_layout_constraint() {
    m_readonly_fwd_called = false;
    auto cn = input(0)->comp_node();
    for (auto i: output())
        if (i->comp_node() != cn) {
            input(0)->add_layout_constraint_contiguous();
            return;
        }
}

void Split::on_mem_status_changed() {
    if (!m_readonly_fwd_called) {
        init_subspec(false);
    }
}

cg::OperatorNodeBase::OprEventCallback
Split::get_opr_event_callback() {
    return {std::bind(&Split::on_mem_status_changed, this)};
}

void Split::on_output_comp_node_stream_changed() {
}

void Split::init_rt_force_dynamic_mem_alloc_imply_chain() {
    auto inp = input(0);
    auto cn0 = inp->comp_node();
    for (auto i: output()) {
        if (i->comp_node() == cn0) {
            i->add_rt_force_dynamic_mem_alloc_imply_chain(inp);
            inp->add_rt_force_dynamic_mem_alloc_imply_chain(i);
        }
    }
}

// f}}}

/* f{{{ ======================= Concat ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Concat);

Concat::Concat(const VarNodeArrayView &inp, int axis,
        const OperatorNodeConfig &config):
    Super{inp[0]->owner_graph(), config, "concat", inp},
    m_axis(axis)
{
    mgb_assert(!inp.empty());
    for (auto &&i : inp) {
        add_input({i});
    }
    add_equivalence_component<ScalarHash<size_t>>(m_axis);
    add_output(None)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

void Concat::get_output_var_shape(
        const TensorShapeArray &inp_shape,
        TensorShapeArray &out_shape) const {

    mgb_assert(inp_shape.size() == input().size());
    mgb_assert(out_shape.size() == 1);
    auto &&oshp = out_shape[0];
    oshp = inp_shape[0];
    mgb_throw_if(m_axis >= static_cast<int>(oshp.ndim) ||
                         m_axis < -static_cast<int>(oshp.ndim),
                 GraphError, "concat axis out of bound: input_ndim=%zu axis=%d",
                 oshp.ndim, m_axis);
    auto real_axis = m_axis;
    if (real_axis < 0)
        real_axis += oshp.ndim;

    for (size_t i = 1; i < inp_shape.size(); ++ i) {
        auto &&tmp = inp_shape[i];
        mgb_throw_if(oshp.ndim != tmp.ndim, GraphError,
                "ndim mismatch: shape=%s inp[%zd]=%s",
                oshp.to_string().c_str(), i, tmp.to_string().c_str());
        for (int n = 0; n < static_cast<int>(tmp.ndim); ++ n) {
            if (n == real_axis) {
                oshp.shape[n] += tmp.shape[n];
            } else {
                mgb_throw_if(oshp.shape[n] != tmp.shape[n], GraphError,
                        "Concat input shapes mismatch: "
                        "accum_out_shape=%s cur_inp_shape=%s inp_idx=%zu"
                        " axis_concat=%d axis_mismatch=%d",
                        oshp.to_string().c_str(), tmp.to_string().c_str(), i,
                        real_axis, n);
            }
        }
    }
}

SymbolVar Concat::make(const VarNodeArrayView& inp, int axis,
                       const OperatorNodeConfig& config) {
    mgb_assert(!inp.empty());
    if (inp.size() == 1)
        return inp[0];
    intl::BatchedDTypePromotion dtp{inp};
    return SymbolVar{inp[0]}.insert_single_output_opr<Concat>(dtp.get_vars(),
                                                              axis, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Concat) {
    auto axis = opr.axis();
    mgb_assert(out_grad.size() == 1);
    OperatorNodeConfig::CompNodeArray comp_node;
    SymbolVarArray partition;
    for (auto i : opr.input()) {
        partition.push_back(GetVarShape::make(i, axis));
        comp_node.push_back(i->comp_node());
    }
    auto ret = Split::make(out_grad[0],
                           Split::Options::make_partition(axis, partition),
                           OperatorNodeConfig().comp_node_arr(comp_node));
    return cg::to_var_node_array(ret);
}
#endif

void Concat::scn_do_execute() {
    auto&& out = output(0)->dev_tensor();
    size_t end = 0;
    for (auto&& input : this->input()) {
        auto&& in = input->dev_tensor();
        auto begin = end;
        auto real_axis = m_axis;
        if (real_axis < 0)
            real_axis += in.shape().ndim;
        end = begin + in.shape().shape[real_axis];
        out.sub(Slice(begin, end).apply(out.layout(), real_axis)).
            copy_from_fixlayout(in);
    }
}

Concat::NodeProp* Concat::do_make_node_prop() const {
    auto rst = Super::do_make_node_prop();
    rst->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    for (auto i: input()) {
        rst->add_dep_type_existing_var(i, NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return rst;
}

void Concat::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();

    using namespace cg::static_infer;

    auto infer_value = [this](
        DeviceTensorND &dest, const InpVal& inp) {

        TensorShape oshp = inp.val[0].shape();
        auto real_axis = m_axis;
        if (real_axis < 0)
            m_axis += oshp.ndim;
        for (size_t i = 1; i < input().size(); ++ i)
            oshp.shape[real_axis] += inp.val.at(i).shape().shape[real_axis];
        dest.resize(oshp);

        size_t end = 0;
        for (size_t i = 0; i < input().size(); ++ i) {
            auto begin = end;
            end = begin + inp.val[i].shape().shape[real_axis];
            dest.sub(Slice(begin, end).apply(dest.layout(), real_axis)).
                copy_from_fixlayout(inp.val[i].value());
        }
        return true;
    };

    DepVal deps;
    for (auto i: input())
        deps.push_back({i, DepType::VALUE});

    owner_graph()->static_infer_manager().register_value_infer(
            output(0),
            {SourceType::DEP, deps, infer_value});
}

void Concat::add_input_layout_constraint() {
    auto cn = output(0)->comp_node();
    for (auto i: input()) {
        if (i->comp_node() != cn) {
            i->add_layout_constraint_contiguous();
        }
    }
}

void Concat::init_output_comp_node() {
    Super::init_output_comp_node();

    auto dcn = output(0)->comp_node();
    for (auto i: input()) {
        if (i->comp_node().mem_node() != dcn.mem_node()) {
            owner_graph()->seq_comp_node_optimizer().register_stream_var(
                    output(0),
                    {CompNode::Stream::COPY,
                     cg::SeqCompNodeOptimizer::StreamPropType::WEAK});
            return;
        }
    }
}

// f}}}

/* f{{{ ======================= ParamPackConcat ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ParamPackConcat);
ParamPackConcat::ParamPackConcat(VarNodeArray& inp, VarNode* table,
                                 const std::vector<dt_int32> offsets_val,
                                 const OperatorNodeConfig& config)
        : Super(inp[0]->owner_graph(), config, "ParamPackConcat", inp),
          m_offsets(offsets_val) {
    CompNode cn = inp[0]->comp_node();
    add_input({inp[0]});
    for (size_t i = 1; i < inp.size(); i++) {
        add_input({inp[i]});
        mgb_assert(cn == inp[i]->comp_node(),
                   "input var for param pack must in same comp node");
    }
    add_input({table});
    add_output(None);
    cg::add_workspace_output(this);

    m_opr = intl::create_megdnn_opr<megdnn::ParamPackConcat>(cn);
}

void ParamPackConcat::add_input_layout_constraint(){
    for (auto i: input()) {
        i->add_layout_constraint_contiguous();
    }
}

SymbolVar ParamPackConcat::make(const SmallVector<SymbolVar>& inp,
                                const SymbolVar& offsets,
                                const std::vector<dt_int32> offsets_val,
                                const OperatorNodeConfig& config) {
    VarNodeArray array(inp.size());
    for (size_t i = 0; i < inp.size(); i++) {
        array[i] = inp[i].node();
    }
    return inp.front().insert_single_output_opr<ParamPackConcat>(
            array, offsets.node(), offsets_val, config);
}

void ParamPackConcat::scn_do_execute() {
    mgb_assert(m_opr.comp_node() == comp_node());
    auto&& inputs = input();
    m_inp_ptr.resize(inputs.size() - 1);
    auto ptr = m_inp_ptr.data();
    for (size_t i = 0; i < inputs.size() - 1; i++) {
        ptr[i] = inputs[i]->dev_tensor().as_megdnn().raw_ptr;
    }
    auto offsets = inputs.back()->dev_tensor().as_megdnn();
    megdnn::TensorND srcs(
            ptr, megdnn::TensorLayout({inputs.size() - 1}, dtype::Int32()));

    auto&& dst = output(0)->dev_tensor().as_megdnn();

    m_opr->exec(srcs, offsets, dst, get_megdnn_workspace_from_var(output(1)));
}

void ParamPackConcat::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

void ParamPackConcat::init_output_static_infer_desc(){
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();

    auto infer_out = [this](TensorShape& dest, const InpVal& inp) {
        dest = {static_cast<unsigned int>(m_offsets.back())};
        return true;
    };
    DepVal shp_deps;
    shp_deps.reserve(input().size());
    for(auto&& inp : input()){
        shp_deps.emplace_back(DepElement{inp, DepType::SHAPE});
    }

    auto infer_wk = [this](TensorShape &dest, const InpVal &inp) {
        TensorShapeArray shapes;
        auto vals = inp.val;
        shapes.reserve(vals.size() - 1);
        for(size_t i = 0; i < vals.size() - 1; i++){
            shapes.push_back(vals[i].shape());
        }
        dest = {m_opr->get_workspace_in_bytes(shapes, vals.back().shape(),
                                              dest)};
        return true;
    };
    mgr.register_shape_infer(output(0), {SourceType::DEP, shp_deps, infer_out});
    mgr.register_shape_infer(output(1), {SourceType::DEP, shp_deps, infer_wk});
}

void ParamPackConcat::on_output_comp_node_stream_changed(){
    Super::on_output_comp_node_stream_changed();
    m_opr = intl::create_megdnn_opr<megdnn::ParamPackConcat>(comp_node());
}
// f}}}

/* f{{{ ======================= ParamPackSplit ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ParamPackSplit);
ParamPackSplit::ParamPackSplit(VarNode* src,
                               const std::vector<dt_int32> offsets,
                               TensorShapeArray& shapes,
                               const OperatorNodeConfig& config)
        : Super{src->owner_graph(), config, "ParamPackSplit", {src}},
          m_shapes(shapes), m_offsets(offsets) {
    add_input({src});

    for (size_t i = 0; i < shapes.size(); i++) {
        mgb_assert(shapes[i].total_nr_elems(), "empty param is not allowed!");
        add_output(ssprintf("param_pack_o%zu", i))
                ->dtype(src->dtype()).shape(shapes[i]);
    }
}

void ParamPackSplit::add_input_layout_constraint(){
    input(0)->add_layout_constraint_contiguous();
}

SymbolVarArray ParamPackSplit::make(const SymbolVar& src,
                                    const std::vector<dt_int32> offsets,
                                    TensorShapeArray shapes,
                                    const OperatorNodeConfig& config) {
    auto&& out = src.node()
                         ->owner_graph()
                         ->insert_opr(std::make_unique<ParamPackSplit>(
                                 src.node(), offsets,
                                 shapes, config))
                         ->output();

    SymbolVarArray ret;
    ret.resize(out.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = out[i];
    }
    return ret;
}

void ParamPackSplit::init_output_dtype() {
    // already initialized in constructor
}

void ParamPackSplit::init_rt_force_dynamic_mem_alloc_imply_chain() {
    for (size_t i = 0; i < output().size(); ++i) {
        auto s = input(0), t = output(i);
        s->add_rt_force_dynamic_mem_alloc_imply_chain(t);
        t->add_rt_force_dynamic_mem_alloc_imply_chain(s);
    }
}

void ParamPackSplit::mem_plan_fwd_in2out_readonly() {
    mgb_assert(m_offsets.size() == output().size() * 2);
    for (size_t i = 0; i < output().size(); i++) {
        auto layout = output(i)->layout();
        auto spec = SubTensorSpec::make_from_offset_elem(layout, m_offsets[i * 2]);
        mgb_assert(output(i)->set_fwd_in2out_readonly(input(0), spec));
    }
}

bool ParamPackSplit::infer_shape(size_t index, TensorShape& dest,
                                 const cg::static_infer::InpVal& inp) {
    dest = m_shapes[index];
    return true;
}

void ParamPackSplit::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    using namespace std::placeholders;
    auto&& mgr = owner_graph()->static_infer_manager();

    for (size_t i = 0; i < output().size(); i++) {
        auto ov = output(i);
        mgr.register_shape_infer(
                ov, {SourceType::CONSTANT, {},
                     std::bind(&ParamPackSplit::infer_shape, this, i, _1, _2)});
    }
}

void ParamPackSplit::scn_do_execute() {
    int inp_size = input(0)->shape().total_nr_elems();
    mgb_assert(inp_size == m_offsets.back(), "input shape should match offsets");
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ParamPackSplit) {
    mgb_assert(out_grad.size() == opr.output().size());
    SmallVector<SymbolVar> grad;
    for (size_t i = 0; i < out_grad.size(); ++i) {
        auto gval = out_grad[i];
        if (!gval) {
            gval = SymbolVar{opr.output(i)}.fill_retain_dtype(0).node();
        }
        grad.emplace_back(gval);
    }
    auto offsets_val = opr.get_offsets();
    auto cn = opr.input(0)->comp_node();
    if (opr.config().has_comp_node_set()) {
        cn = opr.config().get_single_comp_node();
    }
    HostTensorND hv{cn, TensorShape{offsets_val.size()}, dtype::Int32{}};
    memcpy(hv.raw_ptr(), offsets_val.data(), offsets_val.size() * sizeof(int));
    auto offsets = opr::ImmutableTensor::make(*opr.input(0)->owner_graph(), hv);

    return ParamPackConcat::make(
                   grad, offsets, offsets_val,
                   OperatorNodeConfig{}.follow_comp_node(opr.input(0)))
            .node();
}
#endif
// f}}}

/* f{{{ ======================= RelayoutFormat ======================= */
namespace mgb {
namespace opr {
namespace intl {
template <>
struct MegDNNOprInitPostCtor<RelayoutFormat> {
    static void apply(cg::OperatorNodeBase& opr) {
        if (opr.config().output_dtype().valid()) {            
            opr.output(0)->dtype(opr.config().output_dtype());
        } else {
            opr.output(0)->dtype(opr.input(0)->dtype());
        }
    }
};
}  // namespace intl
}  // namespace opr
}  // namespace mgb

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RelayoutFormat);
MEGDNN_OPR_INIT1(RelayoutFormat, "relayout_format")

void RelayoutFormat::init_output_format() {
    TensorFormat src_fmt = input(0)->format(), dst_fmt;
    megdnn_opr()->deduce_format(src_fmt, dst_fmt);
    mgb_assert(output().size() == 2);
    output(0)->format(dst_fmt);
    output(1)->format({});  // default format
}
// f}}}
//
/* f{{{ ===================== WinogradFilterPreprocess ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(WinogradFilterPreprocess);
MEGDNN_OPR_INIT1(WinogradFilterPreprocess, "winograd_filter_preprocess")
void WinogradFilterPreprocess::init_output_dtype() {
    TensorLayout dst;
    TensorLayout src{input(0)->shape(), input(0)->dtype(), input(0)->format()};
    megdnn_opr()->deduce_layout(src, dst);
    output(0)->dtype(dst.dtype);
}
// f}}}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
