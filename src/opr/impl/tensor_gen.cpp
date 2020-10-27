/**
 * \file src/opr/impl/tensor_gen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megdnn/oprs.h"

using namespace mgb;
using namespace opr;

/* ======================= Alloc ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Alloc);

Alloc::Alloc(VarNode* shape, DType dtype, const OperatorNodeConfig &config):
    Super{shape->owner_graph(), config, "alloc", {shape}}
{
    add_input({shape});
    add_output(None)->dtype(dtype);
    outshape_by_symvar_enable(0, 0);
    add_equivalence_component<ScalarHash<const void*>>(dtype.handle());
}

SymbolVar Alloc::make(
        SymbolVar shape, DType dtype, const OperatorNodeConfig &config) {
    return shape.insert_single_output_opr<Alloc>(shape.node(), dtype, config);
}

void Alloc::outshape_by_symvar_do_get_output_shape(
        TensorShape &dest, const ShapeInferInfo &shpinfo) {
    cg::copy_tensor_value_to_shape(dest, *shpinfo.shpval_inp_val.at(0));
}

void Alloc::scn_do_execute() {
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Alloc) {
    MGB_MARK_USED_VAR(wrt_idx);
    MGB_MARK_USED_VAR(out_grad);
    return InvalidGrad::make(opr, 0);
}
#endif

/* ======================= Linspace ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Linspace);

Linspace::Linspace(VarNode* start, VarNode *stop, VarNode *num,
        const Param &param, const OperatorNodeConfig &config):
    Super{start->owner_graph(), config, "linspce", {start, stop}},
    m_param{param}
{
    add_input({start, stop, num});
    add_output(None)->dtype(dtype::Float32());
    add_equivalence_component<PODHash<Param>>(&m_param);
}

SymbolVar Linspace::make(SymbolVar start, SymbolVar stop, SymbolVar num,
        const Param &param, const OperatorNodeConfig &config) {
    return start.insert_single_output_opr<Linspace>(
            start.node(), stop.node(), num.node(), param, config);
}

void Linspace::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_shape = [](TensorShape &dest, const InpVal &inp) {
        cg::copy_tensor_value_to_shape(dest, inp.val[0].value());
        mgb_throw_if(dest.ndim != 1 && !dest.total_nr_elems(), GraphError,
                "Linspace num should contain a scalar; got %s instead",
                dest.to_string().c_str());
        return true;
    };
    mgr.register_shape_infer(output(0),
            {SourceType::DEP, {{input(2), DepType::VALUE}}, infer_shape});
}

cg::OperatorNodeBase::NodeProp* Linspace::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using D = NodeProp::DepType;
    prop->reset_dep_type(input(),
            {D::HOST_VALUE, D::HOST_VALUE, D::HOST_VALUE});
    return prop;
}

void Linspace::scn_do_execute() {
    auto &&mgr = owner_graph()->static_infer_manager();
    auto &&start = mgr.infer_value(input(0)),
         &&stop = mgr.infer_value(input(1));
    mgb_throw_if(!start.shape().is_scalar() || !stop.shape().is_scalar(),
            GraphError,
            "start/stop shape for Linspace must be scalar; get %s %s",
            start.shape().to_string().c_str(),
            stop.shape().to_string().c_str());
    auto startv = DTypeScalar::make_from_raw(
            start.dtype(), start.raw_ptr()).get_cast<double>(),
         stopv = DTypeScalar::make_from_raw(
            stop.dtype(), stop.raw_ptr()).get_cast<double>();

    auto cn = comp_node();
    auto &&opr = m_megdnn_opr;
    if (!opr || opr.comp_node() != cn)
        opr = intl::create_megdnn_opr<megdnn::Linspace>(cn);
    opr->param() = {startv, stopv, m_param.endpoint};
    auto &&ov = output(0)->dev_tensor().as_megdnn();
    mgb_assert(!opr->get_workspace_in_bytes(ov.layout));
    opr->exec(ov, {});
}

void Linspace::record_execute_deps(ExecDependencyArray& deps) {
    deps.emplace_back(
            std::make_unique<intl::MegDNNGraphDep>(std::move(m_megdnn_opr)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Linspace) {
    if (wrt_idx == 2)
        return InvalidGrad::make(opr, wrt_idx);
    mgb_assert(wrt_idx <= 1);
    SymbolVar og{out_grad[0]};
    auto i0 = og.make_scalar(0), i1 = og.make_scalar(1);
    if (!wrt_idx)
        std::swap(i0, i1);
    return opr::Dot::make(og,
            opr::Linspace::make(i0, i1, opr.input(2), opr.param())).node();
}
#endif

/* ======================= Eye ======================= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Eye);

Eye::Eye(VarNode* shape,
        const Param &param, const OperatorNodeConfig &config):
    Super{shape->owner_graph(), config, "eye", {shape}},
    m_param{param}
{
    add_input({shape});
    add_output(None)->dtype(DType::from_enum(param.dtype));
    add_equivalence_component<PODHash<Param>>(&m_param);
}

SymbolVar Eye::make(SymbolVar shape,
        const Param &param, const OperatorNodeConfig &config) {
    return shape.insert_single_output_opr<Eye>(shape.node(), param, config);
}

void Eye::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_shape = [](TensorShape &dest, const InpVal &inp) {
        cg::copy_tensor_value_to_shape(dest, inp.val.at(0).value());
        mgb_throw_if(!dest.ndim || dest.ndim > 2, GraphError,
                "ndim of Eye shape can not exceed 2");
        if (dest.ndim == 1) {
            dest.ndim = 2;
            dest.shape[1] = dest.shape[0];
        }
        return true;
    };
    mgr.register_shape_infer(output(0),
            {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_shape});
}

cg::OperatorNodeBase::NodeProp* Eye::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using D = NodeProp::DepType;
    prop->reset_dep_type(input(), {D::HOST_VALUE});
    return prop;
}

void Eye::scn_do_execute() {
    auto cn = comp_node();
    auto &&opr = m_megdnn_opr;
    if (!opr || opr.comp_node() != cn) {
        opr = intl::create_megdnn_opr<megdnn::Eye>(cn);
        opr->param() = m_param;
    }
    auto &&ov = output(0)->dev_tensor().as_megdnn();
    mgb_assert(!opr->get_workspace_in_bytes(ov.layout));
    opr->exec(ov, {});
}

void Eye::record_execute_deps(ExecDependencyArray& deps) {
    deps.emplace_back(
            std::make_unique<intl::MegDNNGraphDep>(std::move(m_megdnn_opr)));
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Eye) {
    return InvalidGrad::make(opr, wrt_idx);
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

