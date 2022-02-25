#include "megbrain/serialization/sereg.h"
#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_runtime_opr.h"

namespace mgb {
namespace serialization {

template <>
struct OprLoadDumpImpl<opr::TensorRTRuntimeOpr, 0>
        : public opr::TensorRTRuntimeOpr::LoadDumpImpl {};

}  // namespace serialization

namespace opr {
cg::OperatorNodeBase* opr_shallow_copy_tensor_rt_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<TensorRTOpr>();
    return TensorRTOpr::make(
                   opr.trt_builder(), opr.trt_network_def(),
                   opr.trt_graph_feature_bits(), opr.trt_gpu_allocator(),
                   cg::to_symbol_var_array(inputs), opr.trt_cuda_engine(), config)
            .at(0)
            .node()
            ->owner_opr();
}

cg::OperatorNodeBase* opr_shallow_copy_tensor_rt_runtime_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<TensorRTRuntimeOpr>();
    opr.clear_trt_context();
    return TensorRTRuntimeOpr::make(
                   opr.trt_cuda_engine(), opr.trt_gpu_allocator(),
                   cg::to_symbol_var_array(inputs), config)
            .at(0)
            .node()
            ->owner_opr();
}

MGB_REG_OPR_SHALLOW_COPY(TensorRTOpr, opr_shallow_copy_tensor_rt_opr);
MGB_SEREG_OPR(TensorRTRuntimeOpr, 0);
MGB_REG_OPR_SHALLOW_COPY(TensorRTRuntimeOpr, opr_shallow_copy_tensor_rt_runtime_opr);

}  // namespace opr

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
