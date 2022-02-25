#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        PoolingForward, intl::MegDNNOprWrapperFwd<megdnn::PoolingForward>,
        public mixin::AlgoChooserHelper) // {
public:
    MGE_WIN_DECLSPEC_FUC PoolingForward(
            VarNode* src, const Param& param, const ExecutionPolicy& policy,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const ExecutionPolicy& policy = {},
            const OperatorNodeConfig& config = {});

    void init_output_static_infer_desc() override;

    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
};
using Pooling = PoolingForward;

MGB_DEFINE_OPR_CLASS_WITH_EXPORT(
        PoolingBackward, intl::MegDNNOprWrapperBwd<megdnn::PoolingBackward>,
        public mixin::AlgoChooserHelper) // {
public:
    MGE_WIN_DECLSPEC_FUC PoolingBackward(
            VarNode* src, VarNode* dst, VarNode* diff, const Param& param,
            const ExecutionPolicy& policy, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar dst, SymbolVar diff, const Param& param,
            const ExecutionPolicy& policy = {}, const OperatorNodeConfig& config = {});

    MGE_WIN_DECLSPEC_FUC size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override final;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
