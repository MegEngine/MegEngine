#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        SoftmaxForward, intl::MegDNNOprWrapperFwd<megdnn::SoftmaxForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC SoftmaxForward(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param = {},
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

using Softmax = SoftmaxForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        SoftmaxBackward, intl::MegDNNOprWrapperBwd<megdnn::SoftmaxBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC SoftmaxBackward(
            VarNode* x, VarNode* y_grad, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar x, SymbolVar y_grad, const Param& param = {},
            const OperatorNodeConfig& config = {});

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
