/**
 * \file src/opr/impl/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./internal/megdnn_opr_wrapper.inl"

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

namespace mgb {
namespace opr {
namespace intl {
    template<>
    struct MegDNNOprInitPostCtor<Argmax> {
        static void apply(cg::OperatorNodeBase &opr) {
            opr.output(0)->dtype(dtype::Int32());
        }
    };

    template<>
    struct MegDNNOprInitPostCtor<Argmin>: public MegDNNOprInitPostCtor<Argmax> {
    };

    template<>
    struct MegDNNOprInitPostCtor<ArgsortForward> {
        static void apply(cg::OperatorNodeBase &opr) {
            opr.output(0)->dtype(opr.input(0)->dtype());
            opr.output(1)->dtype(dtype::Int32());
        }
    };
}
}
}

/* ================= Argmxx ================= */

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Argmax) {
    MGB_MARK_USED_VAR(out_grad);
    MGB_MARK_USED_VAR(opr);
    mgb_assert(!wrt_idx);
    return nullptr;
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmax);
MEGDNN_OPR_INIT1(Argmax, "argmax")

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Argmin) {
    MGB_MARK_USED_VAR(out_grad);
    MGB_MARK_USED_VAR(opr);
    mgb_assert(!wrt_idx);
    return nullptr;
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Argmin);
MEGDNN_OPR_INIT1(Argmin, "argmin")

/* ================= ArgsortForward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ArgsortForward);
MEGDNN_OPR_CTOR_INIT1(ArgsortForward, "argsort")

std::array<SymbolVar, 2> ArgsortForward::make(
        SymbolVar in_tensor, const Param &param,
        const OperatorNodeConfig &config)
{
    auto node = in_tensor.node()->owner_graph()->insert_opr(
            std::make_unique<ArgsortForward>(in_tensor.node(), param, config));
    mgb_assert(node->output().size() == 3);
    return {node->output(0), node->output(1)};
}

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(ArgsortForward) {
    mgb_assert(out_grad.size() == 3 && wrt_idx == 0 && !out_grad[2]);
    if (!out_grad[0])
        return nullptr;
    return ArgsortBackward::make(out_grad[0], opr.output(1)).node();
}
#endif

/* ================= ArgsortBackward =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ArgsortBackward);
MEGDNN_OPR_INIT3(ArgsortBackward, "argsort_bwd", 2, false)

/* ================= Cumsum =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Cumsum);

Cumsum::Cumsum(VarNode* opr, const Param& param,
               const OperatorNodeConfig& config)
        : Super{opr->owner_graph(), config, "Cumsum", {opr}} {
    init_megdnn_opr(*this, param);
    add_input({opr}, AddInputSortType::CUR_ADDED);
}

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Cumsum) {
    mgb_assert(out_grad[0] && !out_grad[1]);
    auto param = opr.param();
    param.reverse = !param.reverse;
    return Cumsum::make(out_grad[0], param).node();
}
#endif

SymbolVar Cumsum::make(SymbolVar opr, const Param& param,
                       const OperatorNodeConfig& config) {
    return opr.insert_single_output_opr<Cumsum>(opr.node(), param, config);
}

void Cumsum::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output().back()));
}

void Cumsum::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_shape = [](TensorShape& dest, const InpVal& iv) {
        auto ishp = iv.val.at(0).shape();
        dest = ishp;
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
    auto infer_workspace = [this](TensorShape& dest, const InpVal& iv) {
        auto dtype = input(0)->dtype();
        auto ishp = iv.val.at(0).shape();
        TensorLayout ily(ishp, dtype);
        Param real_param = param();
        if (real_param.axis < 0)
            real_param.axis += ishp.ndim;
        megdnn_opr()->param() = real_param;
        dest.ndim = 1;
        dest[0] = megdnn_opr()->get_workspace_in_bytes(ily, ily);
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(1),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_workspace});
}

/* ================= NvOf =================  */

#if MGB_CUDA
MGB_DYN_TYPE_OBJ_FINAL_IMPL(NvOf);

NvOf::NvOf(VarNode* opr, const Param& param, const OperatorNodeConfig& config)
        : Super{opr->owner_graph(), config, "NvOf", {opr}}, m_param{param} {
    constexpr size_t NDIM = 5;
    mgb_assert(opr->dtype() == dtype::Uint8());
    add_input({opr});
    //! NvOf hava only one output
    add_output(None);

    mgb_log_debug("init nvof engine with precision: %u", m_param.precision);
    auto input_shape = this->input()[0]->shape();

    //! nvof input format: nthwc4
    mgb_assert(input_shape.ndim == NDIM);
    //! now only support RGBA format channel data
    mgb_assert(input_shape[4] == 4);

    for (size_t i = 0; i < NDIM; i++) {
        vshape.push_back(input_shape[i]);
    }
}

void NvOf::init_output_dtype() {
    output(0)->dtype(dtype::Int16());
}

SymbolVar NvOf::make(SymbolVar opr, const Param& param,
                     const OperatorNodeConfig& config) {
    return opr.insert_single_output_opr<NvOf>(opr.node(), param, config);
}

void NvOf::scn_do_execute() {
    auto c = this->comp_node();
    //! comp_node may init on CUDA or CPU, eg: lar with --cpu
    //! if ON CUDA, need sync, caused by we use different stream
    if (CompNode::DeviceType::CUDA == c.device_type()) {
        c.sync();
    } else {
        mgb_log_warn(
                "NvOf opr on non CUDA comp_node, which will triger H2D and "
                "D2H!!");
    }

    //! create NvOF engine at same device id of comp_node, can not get
    //! comp_node device id, when NvOf:NvOf, so init at scn_do_execute
    std::lock_guard<std::mutex> lock(m_lock);
    if (init_flag == false) {
        //! nvof sdk do not imp p2p copy, so init nvof engine on the same
        //! device with mgb comp_node
        nv_flow_extractor = std::make_shared<NVFlowExtractor>(
                c.locator().device, vshape, m_param.precision, true, true);
        init_flag = true;
    }

    nv_flow_extractor->extract_flow(
            static_cast<unsigned char*>(
                    input(0)->dev_tensor().as_megdnn().raw_ptr),
            vshape,
            reinterpret_cast<int16_t*>(
                    output(0)->dev_tensor().as_megdnn().raw_ptr));
}

void NvOf::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_shape = [](TensorShape& dest, const InpVal& iv) {
        auto ishp = iv.val.at(0).shape();
        SmallVector<size_t> tv;
        tv.push_back(ishp[0]);
        tv.push_back(ishp[1] - 1);
        tv.push_back(ishp[2] / 4);
        tv.push_back(ishp[3] / 4);
        tv.push_back(ishp[4] / 2);
        dest = tv;

        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_shape});
}
#endif

/* ================= CondTake =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondTake);

CondTake::CondTake(VarNode *data, VarNode *mask,
        const Param &param, const OperatorNodeConfig &config):
    Super(data->owner_graph(), config, "cond_take", {data, mask})
{
    init_megdnn_opr(*this, param);
    add_input({data, mask});
    auto dtypes = megdnn_opr()->infer_dtype(data->dtype(), mask->dtype());
    for (int i = 0; i < 2; ++ i) {
        output(i)
            ->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
            .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
            .dtype(dtypes[i]);
    }
}

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CondTake) {
    mgb_assert(out_grad.size() == 3 && !out_grad[2]);
    if (wrt_idx == 0 && out_grad[0]) {
        SymbolVar data_sym{opr.input(0)};
        auto inp_set = IndexingIncrMultiAxisVec::make(
                data_sym.flatten().fill_retain_dtype(0), out_grad[0],
                {indexing::AxisIndexer::make_index(0, opr.output(1))});
        return inp_set.reshape(data_sym.symshape()).node();
    }
    return nullptr;
}
#endif

std::array<SymbolVar, 2> CondTake::make(
        SymbolVar data, SymbolVar mask,
        const Param &param, const OperatorNodeConfig &config) {
    auto ov0 = data.insert_single_output_opr<CondTake>(
            data.node(), mask.node(), param, config);
    return {ov0, ov0.node()->owner_opr()->output(1)};
}

void CondTake::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto infer_workspace = [this](TensorShape& dest, const InpVal& iv) {
        auto dtype = input(0)->dtype();
        TensorLayout ily(iv.val[0].shape(), dtype);
        dest.ndim = 1;
        dest.shape[0] = megdnn_opr()->get_workspace_in_bytes(ily);
        return true;
    };
    owner_graph()->static_infer_manager().register_shape_infer(
            output(2),
            {SourceType::DEP, {{input(0), DepType::SHAPE}}, infer_workspace});
}

void CondTake::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void CondTake::scn_do_execute() {
    intl::MegDNNDynOutMallocImpl dyn_malloc{this, comp_node()};
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(1)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output().back()),
                       &dyn_malloc);
}

/* ================= TopK =================  */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(TopK);

TopK::TopK(VarNode* data, VarNode* k, const Param& param,
           const OperatorNodeConfig& config)
        : Super(data->owner_graph(), config, "top_k", {data, k}) {
    init_megdnn_opr(*this, param);
    add_input({data, k});
    if (param.mode == Param::Mode::KTH_ONLY) {
        output(1)
                ->add_flag(VarNode::Flag::VOLATILE_CONTENT)
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
}

std::array<SymbolVar, 2> TopK::make(SymbolVar data, SymbolVar k,
                                    const Param& param,
                                    const OperatorNodeConfig& config) {
    auto opr = data.node()->owner_graph()->insert_opr(
            std::make_unique<TopK>(data.node(), k.node(), param, config));
    auto o1 = opr->output(1);
    if (param.mode == Param::Mode::KTH_ONLY) {
        o1 = nullptr;
    }
    return {opr->output(0), o1};
}

void TopK::init_output_dtype() {
    mgb_assert(input(1)->dtype() == dtype::Int32{}, "k must be int32, got %s",
               input(1)->dtype().name());
    output(0)->dtype(input(0)->dtype());
    output(1)->dtype(dtype::Int32{});
}

void TopK::add_input_layout_constraint() {
    auto check = [](const TensorLayout& layout) {
        mgb_assert(layout.ndim == 2, "top-k input must be two-dim, got %s",
                   layout.TensorShape::to_string().c_str());
        return layout.stride[1] == 1;
    };
    input(0)->add_layout_constraint(check);
}

void TopK::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    auto infer_oshp0 = [this](TensorShape& dst, const InpVal& iv) {
        auto&& k_tensor = iv.val[1].value();
        mgb_assert(k_tensor.shape().is_scalar(), "k must be scalar, got %s",
                   k_tensor.shape().to_string().c_str());
        TensorLayout o0, o1;
        megdnn_opr()->deduce_layout(k_tensor.ptr<int>()[0],
                                    {iv.val[0].shape(), input(0)->dtype()}, o0,
                                    o1);
        dst = o0;
        return true;
    };
    mgr.register_shape_infer(output(0), {SourceType::DEP,
                                         {{input(0), DepType::SHAPE},
                                          {input(1), DepType::VALUE}},
                                         infer_oshp0});

    if (param().mode == Param::Mode::KTH_ONLY) {
        mgr.register_shape_infer(output(1), ShapeInferDesc::make_const({}));
    } else {
        mgr.register_shape_infer(output(1),
                                 ShapeInferDesc::make_identity(output(0)));
    }

    auto infer_workspace = [this](TensorShape& dst, const InpVal& iv) {
        auto k = iv.val[3].value().ptr<int>()[0];
        auto size = megdnn_opr()->get_workspace_in_bytes(
                k, {iv.val[0].shape(), input(0)->dtype()},
                {iv.val[1].shape(), output(0)->dtype()},
                {iv.val[2].shape(), output(1)->dtype()});
        dst.ndim = 1;
        dst.shape[0] = size;
        return true;
    };
    mgr.register_shape_infer(output(2), {SourceType::DEP,
                                         {{input(0), DepType::SHAPE},
                                          {output(0), DepType::SHAPE},
                                          {output(1), DepType::SHAPE},
                                          {input(1), DepType::VALUE}},
                                         infer_workspace});
}

void TopK::scn_do_execute() {
    auto&& mgr = owner_graph()->static_infer_manager();
    auto k = mgr.infer_value(input(1)).ptr<int>()[0];
    megdnn_opr()->exec(k, input(0)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       output(1)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output(2)));
}

void TopK::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

#ifdef MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(TopK) {
    if (opr.param().mode == TopK::Param::Mode::KTH_ONLY) {
        mgb_assert(out_grad[0] && !out_grad[1] && !out_grad[2]);
        auto add_axis = [](SymbolVar x) {
            return opr::AxisAddRemove::make(
                    x, {opr::AxisAddRemove::AxisDesc::make_add(1)});
        };
        SymbolVar mask = opr::eq(add_axis(opr.output(0)), opr.input(0)),
                  og = add_axis(out_grad[0]) / opr::reduce_ax_sum(mask, 1);
        return (og * mask).node();
    }
    if (!out_grad[0])
        return nullptr;
    return ArgsortBackward::make(out_grad[0], opr.output(1), opr.input(0))
            .node();
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
