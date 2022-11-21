#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        InstanceNormForward, intl::MegDNNOprWrapperFwd<megdnn::GroupNormForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC InstanceNormForward(
            VarNode* data, VarNode* weight, VarNode* bias, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC InstanceNormForward(
            VarNode* data, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar data, SymbolVar weight, SymbolVar bias, const Param& param = {},
            const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar data, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void scn_do_execute() override;
};
using InstanceNorm = InstanceNormForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        InstanceNormBackward, intl::MegDNNOprWrapperBwd<megdnn::GroupNormBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC InstanceNormBackward(
            VarNode* diff, VarNode* data, VarNode* weight, VarNode* mean, VarNode* rstd,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC InstanceNormBackward(
            VarNode* diff, VarNode* data, VarNode* mean, VarNode* rstd,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar diff, SymbolVar data, SymbolVar weight, SymbolVar mean,
            SymbolVar rstd, const Param& param = {},
            const OperatorNodeConfig& config = {});
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar diff, SymbolVar data, SymbolVar mean, SymbolVar rstd,
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

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
