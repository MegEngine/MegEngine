/**
 * \file src/opr/include/megbrain/opr/rand.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

namespace intl {

template<typename MegDNNOpr>
MGB_DEFINE_CLS_WITH_SUPER(RNGOprBase, cg::SingleCNOperatorNodeBase) // {
    public:
        using Param = typename MegDNNOpr::Param;
        const Param& param() const {
            return m_param;
        }

    private:
        Param m_param;
        UniqPtrWithCN<MegDNNOpr> create_megdnn_opr();

    protected:
        ~RNGOprBase(){};
        RNGOprBase(const OperatorNodeBaseCtorParam &opr, const Param &param);
        void ensure_megdnn_opr();
        UniqPtrWithCN<MegDNNOpr> m_dnn_opr;
};

/* ================= RNG with shape =================  */
#define _DEFINE_RNG_OPR_WITH_SHAPE_CLASS(RNG)                                  \
    MGB_DEFINE_OPR_CLASS(RNG, RNGOprBase<megdnn::RNG>)                         \
    cg::OperatorNodeBase::NodeProp* do_make_node_prop() const override;        \
                                                                               \
public:                                                                        \
    RNG(VarNode* shape, const Param& param, const OperatorNodeConfig& config); \
    static SymbolVar make(SymbolVar shape, const Param& param = {},            \
                          const OperatorNodeConfig& config = {});              \
    static SymbolVar make(ComputingGraph& graph, const TensorShape& shape,     \
                          const OperatorNodeConfig& config,                    \
                          const Param& param = {}) {                           \
        return make(var_from_tensor_shape(graph, config, "rng", shape), param, \
                    config);                                                   \
    }                                                                          \
    void init_output_static_infer_desc() override;                             \
    void scn_do_execute() override;                                            \
    }                                                                          \
    ;

_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(UniformRNG)
_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(GaussianRNG)
_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(PermutationRNG)
#undef _DEFINE_RNG_OPR_WITH_SHAPE_CLASS

/* ================= RNG with input =================  */
#define _DEFINE_RNG_OPR_WITH_INPUT_CLASS(RNG)                                      \
MGB_DEFINE_OPR_CLASS(RNG, RNGOprBase<megdnn::RNG>)                                 \
    void add_input_layout_constraint() override;                                   \
    cg::OperatorNodeBase::NodeProp* do_make_node_prop() const override;            \
    public:                                                                        \
        RNG(_INPUTS(VarNode*), const Param &param,                                 \
            const OperatorNodeConfig &config);                                     \
        static _OUTPUTS make(_INPUTS(SymbolVar),const Param &param = {},           \
                const OperatorNodeConfig &config = {});                            \
        void init_output_static_infer_desc() override;                             \
        void scn_do_execute() override;                                            \
};

/* ================= 1 input =================  */
#define _INPUTS(preifx) preifx i0
#define _OUTPUTS SymbolVar
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(PoissonRNG)
#undef _OUTPUTS
#define _OUTPUTS SymbolVarArray
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(ShuffleRNGForward)
#undef _OUTPUTS
#undef _INPUTS

/* ================= 2 input =================  */
#define _INPUTS(preifx) preifx i0, preifx i1
#define _OUTPUTS SymbolVar
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(BetaRNG)
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(GammaRNG)
#undef _OUTPUTS
#undef _INPUTS
#undef _DEFINE_RNG_OPR_WITH_INPUT_CLASS

}  // intl

using UniformRNG = intl::UniformRNG;
using GaussianRNG = intl::GaussianRNG;
using GammaRNG = intl::GammaRNG;
using PermutationRNG = intl::PermutationRNG;
using PoissonRNG = intl::PoissonRNG;
using BetaRNG = intl::BetaRNG;
using ShuffleRNG = intl::ShuffleRNGForward;

MGB_DEFINE_OPR_CLASS(ShuffleRNGBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::ShuffleRNGBackward>)  //{
public:
ShuffleRNGBackward(VarNode* out_diff, VarNode* indices, VarNode* result_shape,
                   const Param& param, const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar out_diff, SymbolVar indices,
                      SymbolVar result_shape, const Param& param = {},
                      const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
