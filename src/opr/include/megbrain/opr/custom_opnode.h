#pragma once

#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include "megbrain/custom/adaptor.h"
#include "megbrain/custom/custom.h"
#include "megbrain/custom/manager.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/graph/symbol_var.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace opr {

using VarNode = cg::VarNode;
using VarNodeArray = cg::VarNodeArray;
using SymbolVar = cg::SymbolVar;
using SymbolVarArray = cg::SymbolVarArray;
using StaticInferInpVal = cg::StaticInferInpVal;
using OperatorNodeConfig = cg::OperatorNodeConfig;

MGB_DEFINE_OPR_CLASS(CustomOpNode, cg::OperatorNodeBase) // {
    const std::shared_ptr<const custom::CustomOp> m_op;
    custom::Param m_param;
    CompNode m_comp_node;
    TensorShapeArray m_out_shape;

    void infer_output_comp_node(void);
    void infer_output_dtype(void);
    void infer_output_format(void);
    void infer_output_shape(void);
    void infer_output_shape(
            const TensorShapeArray& input_shapes, TensorShapeArray& output_shapes);

    // called by computing_graph for each output varnode
    bool infer_desc(
            size_t out_idx, TensorShape& output_shape,
            const StaticInferInpVal& input_vals);

    void init_output_dtype() override final;
    void init_output_format() override final;
    void init_output_comp_node() override final;
    void do_execute(ExecEnv& env) override final;
    void init_output_static_infer_desc() override final;
    void init_output_mem_plan(bool dynamic) override final;

    // [TODO] if some dynamic mem alloc flag in m_opimpl, ignore it for now
    void init_rt_force_dynamic_mem_alloc_imply_chain() override final;

    // [TODO] only contiguous input is supported
    void add_input_layout_constraint() override final;

    // [TODO] ignore it for now
    void mem_plan_fwd_in2out_readonly() override final;

    // [TODO] ignore it for now
    void mem_plan_fwd_in2out_writable() override final;

    // [TODO] return default ctor obj
    OprEventCallback get_opr_event_callback() override final;

    // [TODO]
    void on_output_comp_node_stream_changed() override final;

    // [TODO]
    NodeProp* do_make_node_prop() const override final;

    // [TODO] default implementation
    bool update_priority() const override final;

public:
    MGE_WIN_DECLSPEC_FUC CustomOpNode(
            const std::shared_ptr<const custom::CustomOp>& op, VarNodeArray inputs,
            const custom::Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static VarNodeArray make(
            const std::shared_ptr<const custom::CustomOp>& op, VarNodeArray inputs,
            const custom::Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            const std::shared_ptr<const custom::CustomOp>& op, SymbolVarArray inputs,
            const custom::Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC custom::RunTimeId runtime_id(void) const;
    MGE_WIN_DECLSPEC_FUC uint32_t param_tag(void) const;
    MGE_WIN_DECLSPEC_FUC custom::Param& param(void);
    MGE_WIN_DECLSPEC_FUC custom::Param param(void) const;
    MGE_WIN_DECLSPEC_FUC std::string op_type(void) const;
    MGE_WIN_DECLSPEC_FUC std::string op_desc(void) const;
    MGE_WIN_DECLSPEC_FUC size_t input_num(void) const;
    MGE_WIN_DECLSPEC_FUC size_t output_num(void) const;
    MGE_WIN_DECLSPEC_FUC custom::ArgInfo input_info(size_t idx) const;
    MGE_WIN_DECLSPEC_FUC custom::ArgInfo output_info(size_t idx) const;
};

}  // namespace opr
}  // namespace mgb

#endif
