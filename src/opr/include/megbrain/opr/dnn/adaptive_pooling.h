#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(
        AdaptivePoolingForward,
        intl::WorkspaceSizeInfer<intl::OutshapeBySymvarSCNOpr<
                mixin::MegDNNOprHolderImpl<megdnn::AdaptivePoolingForward>>>) // {
public:
    MGE_WIN_DECLSPEC_FUC AdaptivePoolingForward(
            VarNode* src, VarNode* out_shape, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar out_shape, const Param& param,
            const OperatorNodeConfig& config = {});
    static SymbolVar make(
            SymbolVar src, const TensorShape& out_shape, const Param& param,
            const OperatorNodeConfig& config = {}) {
        return make(src, cg::var_from_tensor_shape(src, out_shape), param, config);
    }

private:
    void scn_do_execute() override;
    void outshape_by_symvar_do_get_output_shape(
            TensorShape& dest, const ShapeInferInfo& shpinfo) override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
    void init_output_dtype() override;
    void add_input_layout_constraint() override;
    void init_output_static_infer_desc() override;
    void record_execute_deps(ExecDependencyArray& deps) override;
    NodeProp* do_make_node_prop() const override;
};
using AdaptivePooling = AdaptivePoolingForward;

MGB_DEFINE_OPR_CLASS(
        AdaptivePoolingBackward,
        intl::MegDNNOprWrapperBwd<megdnn::AdaptivePoolingBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC AdaptivePoolingBackward(
            VarNode* src, VarNode* out_shape, VarNode* dst, VarNode* diff,
            const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar out_shape, SymbolVar dst, SymbolVar diff,
            const Param& param, const OperatorNodeConfig& config = {});

private:
    void scn_do_execute() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
