/**
 * \file src/opr/impl/rand.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/rand.h"
#include "megbrain/opr/utility.h"
#include "megbrain/graph/grad_impl.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace intl;

namespace {


template<class MegDNNOpr>
struct RNGName;

template<>
struct RNGName<megdnn::UniformRNG> {
    static constexpr const char* name = "uniform_rng";
};

template<>
struct RNGName<megdnn::GaussianRNG> {
    static constexpr const char* name = "gaussian_rng";
};

} // anonymous namespace

RNGOprBase::RNGOprBase(const OperatorNodeBaseCtorParam &opr, VarNode *shape):
    Super(opr)
{
    add_input({shape});
    add_output(None)->dtype(dtype::Float32());
    cg::add_workspace_output(this);

    // disable dedup
    add_equivalence_component<ScalarHash<void*>>(this);
}

RNGOprBase::~RNGOprBase() {
}

cg::OperatorNodeBase::NodeProp* RNGOprBase::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::IMPURE_FUNC);
    prop->reset_dep_type(input(), {NodeProp::DepType::HOST_VALUE});
    return prop;
}

void RNGOprBase::ensure_megdnn_opr() {
    if (!m_megdnn_opr || m_megdnn_opr.comp_node() != comp_node()) {
        // activate comp_node for curandCreateGenerator in create_megdnn_opr
        comp_node().activate();
        m_megdnn_opr = create_megdnn_opr();
    }
}

void RNGOprBase::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto infer_out = [](TensorShape &dest, const InpVal &inp) {
        cg::copy_tensor_value_to_shape(dest, inp.val.at(0).value());
        return true;
    };
    auto infer_wk = [this](TensorShape &dest, const InpVal &inp) {
        ensure_megdnn_opr();
        dest.ndim = 1;
        dest.shape[0] = m_megdnn_opr->get_workspace_in_bytes(
                {inp.val.at(0).shape(), output(0)->dtype()});
        return true;
    };
    mgr.register_shape_infer(output(0),
            {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_out});
    mgr.register_shape_infer(output(1),
            {SourceType::DEP, {{output(0), DepType::SHAPE}}, infer_wk});
}

void RNGOprBase::scn_do_execute() {
    m_megdnn_opr->exec(
            output(0)->dev_tensor().as_megdnn(),
            get_megdnn_workspace_from_var(output(1)));
}

template<class MegDNNOpr>
RNGOpr<MegDNNOpr>::RNGOpr(VarNode *shape, const Param &param,
        const OperatorNodeConfig &config):
    Super({shape->owner_graph(), config, RNGName<MegDNNOpr>::name, {shape}},
            shape),
    m_param(param)
{
}

template<class MegDNNOpr>
SymbolVar RNGOpr<MegDNNOpr>::make(SymbolVar shape, const Param &param,
        const OperatorNodeConfig &config) {
    return shape.insert_single_output_opr<RNGOpr>(shape.node(), param, config);
}

template<class MegDNNOpr>
UniqPtrWithCN<megdnn::RNGBase> RNGOpr<MegDNNOpr>::create_megdnn_opr() {
    auto opr = intl::create_megdnn_opr<MegDNNOpr>(comp_node());
    opr->param() = param();
    return opr;
}

#define IMPL(_cls)                                      \
    MGB_IMPL_OPR_GRAD(_cls) {                           \
        MGB_MARK_USED_VAR(out_grad);                    \
        return InvalidGrad::make(opr, wrt_idx);         \
    }

namespace mgb {
namespace opr {
namespace intl {
template class RNGOpr<::megdnn::GaussianRNG>;
template class RNGOpr<::megdnn::UniformRNG>;
#if MGB_ENABLE_GRAD
IMPL(GaussianRNG);
IMPL(UniformRNG);
#endif
}
}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

