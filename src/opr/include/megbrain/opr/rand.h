#pragma once

#include "megbrain/graph.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

namespace intl {

template <typename MegDNNOpr>
MGB_DEFINE_CLS_WITH_SUPER(RNGOprBase, cg::SingleCNOperatorNodeBase) // {
public:
    using Param = typename MegDNNOpr::Param;
    const Param& param() const { return m_param; }

private:
    Param m_param;
    UniqPtrWithCN<MegDNNOpr> create_megdnn_opr();

protected:
    ~RNGOprBase(){};
    RNGOprBase(const OperatorNodeBaseCtorParam& opr, const Param& param);
    void ensure_megdnn_opr();
    UniqPtrWithCN<MegDNNOpr> m_dnn_opr;
};

/* ================= RNG with shape =================  */
#define _DEFINE_RNG_OPR_WITH_SHAPE_CLASS(RNG)                                      \
    MGB_DEFINE_OPR_CLASS_WITH_EXPORT(RNG, RNGOprBase<megdnn::RNG>)                 \
        cg::OperatorNodeBase::NodeProp* do_make_node_prop() const override;        \
                                                                                   \
    public:                                                                        \
        RNG(VarNode* shape, const Param& param, const OperatorNodeConfig& config); \
        MGE_WIN_DECLSPEC_FUC static SymbolVar make(                                \
                SymbolVar shape, const Param& param = {},                          \
                const OperatorNodeConfig& config = {});                            \
        static SymbolVar make(                                                     \
                ComputingGraph& graph, const TensorShape& shape,                   \
                const OperatorNodeConfig& config, const Param& param = {}) {       \
            return make(                                                           \
                    var_from_tensor_shape(graph, config, "rng", shape), param,     \
                    config);                                                       \
        }                                                                          \
        void init_output_static_infer_desc() override;                             \
        void scn_do_execute() override;                                            \
    };

_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(UniformRNG)
_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(GaussianRNG)
_DEFINE_RNG_OPR_WITH_SHAPE_CLASS(PermutationRNG)
#undef _DEFINE_RNG_OPR_WITH_SHAPE_CLASS

/* ================= RNG with input =================  */
#define _DEFINE_RNG_OPR_WITH_INPUT_CLASS(RNG)                                         \
    MGB_DEFINE_OPR_CLASS_WITH_EXPORT(RNG, RNGOprBase<megdnn::RNG>)                    \
        void add_input_layout_constraint() override;                                  \
        cg::OperatorNodeBase::NodeProp* do_make_node_prop() const override;           \
                                                                                      \
    public:                                                                           \
        RNG(_INPUTS(VarNode*), const Param& param, const OperatorNodeConfig& config); \
        MGE_WIN_DECLSPEC_FUC static _OUTPUTS make(                                    \
                _INPUTS(SymbolVar), const Param& param = {},                          \
                const OperatorNodeConfig& config = {});                               \
        void init_output_static_infer_desc() override;                                \
        void scn_do_execute() override;                                               \
    };

/* ================= 1 input =================  */
#define _INPUTS(preifx) preifx i0
#define _OUTPUTS        SymbolVar
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(PoissonRNG)
#undef _OUTPUTS
#define _OUTPUTS SymbolVarArray
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(ShuffleRNGForward)
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(DropoutForward)
#undef _OUTPUTS
#undef _INPUTS

/* ================= 2 input =================  */
#define _INPUTS(preifx) preifx i0, preifx i1
#define _OUTPUTS        SymbolVar
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(BetaRNG)
_DEFINE_RNG_OPR_WITH_INPUT_CLASS(GammaRNG)
#undef _OUTPUTS
#undef _INPUTS

#undef _DEFINE_RNG_OPR_WITH_INPUT_CLASS

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        MultiHeadAttnForward, RNGOprBase<megdnn::MultiHeadAttnForward>) // {
    void add_input_layout_constraint() override;
    cg::OperatorNodeBase::NodeProp* do_make_node_prop() const override;

public:
    MultiHeadAttnForward(
            VarNode* queries, VarNode* keys, VarNode* values, VarNode* qkvo_weight_bias,
            VarNode* attn_mask, VarNode* bias_k, VarNode* bias_v, const Param& param,
            const OperatorNodeConfig& config);
    MultiHeadAttnForward(
            VarNode* queries, VarNode* keys, VarNode* values, VarNode* qkvo_weight_bias,
            VarNode* attn_mask, const Param& param, const OperatorNodeConfig& config);
    MultiHeadAttnForward(
            VarNode* queries, VarNode* keys, VarNode* values, VarNode* qkvo_weight_bias,
            VarNode* bias_k, VarNode* bias_v, const Param& param,
            const OperatorNodeConfig& config);
    MultiHeadAttnForward(
            VarNode* queries, VarNode* keys, VarNode* values, VarNode* qkvo_weight_bias,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, SymbolVar attn_mask, SymbolVar bias_k,
            SymbolVar bias_v, const Param& param = {},
            const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, SymbolVar attn_mask, const Param& param = {},
            const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, SymbolVar bias_k, SymbolVar bias_v,
            const Param& param = {}, const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, const Param& param = {},
            const OperatorNodeConfig& config = {});
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
};

}  // namespace intl

using UniformRNG = intl::UniformRNG;
using GaussianRNG = intl::GaussianRNG;
using GammaRNG = intl::GammaRNG;
using PermutationRNG = intl::PermutationRNG;
using PoissonRNG = intl::PoissonRNG;
using BetaRNG = intl::BetaRNG;
using ShuffleRNG = intl::ShuffleRNGForward;
using Dropout = intl::DropoutForward;
using DropoutForward = intl::DropoutForward;
using MultiHeadAttn = intl::MultiHeadAttnForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        ShuffleRNGBackward, intl::MegDNNOprWrapperBwd<megdnn::ShuffleRNGBackward>) // {
public:
    ShuffleRNGBackward(
            VarNode* out_diff, VarNode* indices, VarNode* result_shape,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar out_diff, SymbolVar indices, SymbolVar result_shape,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        DropoutBackward, intl::MegDNNOprWrapperBwd<megdnn::DropoutBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC DropoutBackward(
            VarNode* doup, VarNode* mask, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar doup, SymbolVar mask, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void scn_do_execute() override;
};

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        MultiHeadAttnBackward,
        intl::MegDNNOprWrapperBwd<megdnn::MultiHeadAttnBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC MultiHeadAttnBackward(
            VarNode* diff, VarNode* queries, VarNode* keys, VarNode* values,
            VarNode* qkvo_weight_bias, VarNode* attn_mask, VarNode* attn_weight,
            VarNode* mask_reservespace, VarNode* othr_reservespace, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC MultiHeadAttnBackward(
            VarNode* diff, VarNode* queries, VarNode* keys, VarNode* values,
            VarNode* qkvo_weight_bias, VarNode* attn_weight, VarNode* mask_reservespace,
            VarNode* othr_reservespace, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar diff, SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, SymbolVar attn_mask, SymbolVar attn_weight,
            SymbolVar mask_reservespace, SymbolVar othr_reservespace,
            const Param& param = {}, const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar diff, SymbolVar queries, SymbolVar keys, SymbolVar values,
            SymbolVar qkvo_weight_bias, SymbolVar attn_weight,
            SymbolVar mask_reservespace, SymbolVar othr_reservespace,
            const Param& param = {}, const OperatorNodeConfig& config = {});

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void scn_do_execute() override;
};

}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
