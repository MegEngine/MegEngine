/**
 * \file src/opr/include/megbrain/opr/rand.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

namespace intl {

MGB_DEFINE_CLS_WITH_SUPER(RNGOprBase, cg::SingleCNOperatorNodeBase) // {
    UniqPtrWithCN<megdnn::RNGBase> m_megdnn_opr;

    void ensure_megdnn_opr();
    void init_output_static_infer_desc() override;
    void scn_do_execute() override final;

    protected:
        RNGOprBase(const OperatorNodeBaseCtorParam &opr, VarNode *shape);
        ~RNGOprBase();
        NodeProp* do_make_node_prop() const override;

        virtual UniqPtrWithCN<megdnn::RNGBase> create_megdnn_opr() = 0;
};

template<class MegDNNOpr>
MGB_DEFINE_OPR_CLASS(RNGOpr, RNGOprBase) // {

    public:
        using Param = typename MegDNNOpr::Param;

        RNGOpr(VarNode *shape, const Param &param,
                const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar shape, const Param &param = {},
                const OperatorNodeConfig &config = {});

        static SymbolVar make(ComputingGraph &graph, const TensorShape &shape,
                const OperatorNodeConfig &config,
                const Param &param = {}) {
            return make(var_from_tensor_shape(graph, config, "rng", shape),
                    param, config);
        }

        const Param& param() const {
            return m_param;
        }

    private:
        Param m_param;
        UniqPtrWithCN<megdnn::RNGBase> create_megdnn_opr() override;
};

#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL template<class MegDNNOpr>
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNGOpr<MegDNNOpr>);
#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL

} // intl

using UniformRNG = intl::RNGOpr<megdnn::UniformRNG>;
using GaussianRNG = intl::RNGOpr<megdnn::GaussianRNG>;

} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

