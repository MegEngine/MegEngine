/**
 * \file src/opr/include/megbrain/opr/cond.h
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
#include "megbrain/graph/execution_mask.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs/general.h"

#if MGB_ENABLE_COND_EXEC

namespace mgb {
namespace opr {

/*!
 * \brief Evaluate the predicate and generate ExecutionMask with PPV.
 *
 * This opr would setup execution masks for the branches according to the value
 * of the predicate var, values of keys and operation mode.
 *
 * The actual inputs of the operator in the graph are keys + [pred]. The outputs
 * are PPVs corresponding to the branches.
 */
MGB_DEFINE_OPR_CLASS(CondExecPred, cg::SingleCNOperatorNodeBase) // {
public:
    using ExecutionMask = cg::ExecutionMask;
    using Param = megdnn::param::CondExecPred;
    using ExecutionMaskArray = SmallVector<std::shared_ptr<ExecutionMask>>;
    using Mode = Param::Mode;

    //! global registry on the graph to handle propogation of ExecutionMask and
    //! related runtime management
    class GlobalRegistry;

    CondExecPred(VarNode* pred, const VarNodeArrayView& keys,
                 const Param& param, const OperatorNodeConfig& config);

    static OperatorNodeBase* make_opr(SymbolVar pred,
                                      const VarNodeArrayView& keys,
                                      const Param& param,
                                      const OperatorNodeConfig& config);

    static SymbolVarArray make(SymbolVar pred, const VarNodeArrayView& keys,
                               const Param& param = {},
                               const OperatorNodeConfig& config = {}) {
        return cg::to_symbol_var_array(
                make_opr(pred, keys, param, config)->output());
    }

    const ExecutionMaskArray& masks() const { return m_masks; }

    VarNode* out_var_from_mask(ExecutionMask* mask) const;

    const Param& param() const { return m_param; }

private:
    //! compare predicate and branch keys
    class PredEvaluator;

    const Param m_param;

    ExecutionMaskArray m_masks;

    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};

/*!
 * \brief Compute a logical function over a set of PPVs.
 *
 * This is primarily used by SUM_COND_OUT mode of CondExecMerge. This opr serves
 * the purpose of ExecutionMask deduplication, which is important because we
 * require all inputs of an operator to be controlled by the same ExecutionMask.
 *
 * All the inputs must be PPVs and this opr also produces a PPV.
 *
 * Note: this opr can be used for copying a PPV to another comp node by
 * specifying the target comp node in the config.
 */
MGB_DEFINE_OPR_CLASS(CondExecPredLogical, cg::SingleCNOperatorNodeBase) // {
public:
    using Param = megdnn::param::CondExecPredLogical;
    using Mode = Param::Mode;
    using ExecutionMask = cg::ExecutionMask;

    CondExecPredLogical(const VarNodeArrayView& preds, const Param& param,
                        const OperatorNodeConfig& config);

    static SymbolVar make(const VarNodeArrayView& preds, const Param& param,
                          const OperatorNodeConfig& config = {});

    ExecutionMask* mask() const { return m_mask.get(); }

    const Param& param() const { return m_param; }

    static const char* mode2str(Mode mode);

private:
    //! compute the logical function
    class PredEvaluator;

    SmallVector<ExecutionMask*> m_input_masks;
    std::shared_ptr<ExecutionMask> m_mask;

    const Param m_param;

    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};

/*!
 * \brief Mark the beginning of conditional execution.
 *
 * This operator would forward all the inputs to all the outputs.
 *
 * The actual opr inputs in the graph are inputs + [ppv]
 */
MGB_DEFINE_OPR_CLASS(CondExecMark, cg::SingleCNOperatorNodeBase) // {
public:
    using Param = megdnn::param::CondExecMark;

    CondExecMark(VarNode* ppv, const VarNodeArrayView& inputs,
                 const Param& param, const OperatorNodeConfig& config);

    static OperatorNodeBase* make_opr(SymbolVar ppv,
                                      const VarNodeArrayView& inputs,
                                      const Param& param,
                                      const OperatorNodeConfig& config);

    static SymbolVarArray make(SymbolVar ppv, const VarNodeArrayView& inputs,
                               const Param& param = {},
                               const OperatorNodeConfig& config = {}) {
        return cg::to_symbol_var_array(
                make_opr(ppv, inputs, param, config)->output());
    }

    /*!
     * \brief mark the input var by a CondExecMark if \p maybe_ppv is a PPV or
     *      controlled by ExecutionMask
     */
    static SymbolVar mark_if_need(SymbolVar maybe_ppv, SymbolVar input,
                                  const Param& param = {},
                                  const OperatorNodeConfig& config = {});

    const Param& param() const { return m_param; }

    //! whether shape inference is disabled in the param
    bool has_no_shape_infer() const {
        return m_param.static_infer == Param::StaticInfer::NONE;
    }

private:
    const Param m_param;
    std::vector<bool> m_mem_fwd_success;

    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void init_rt_force_dynamic_mem_alloc_imply_chain() override;
    void mem_plan_fwd_in2out_readonly() override;
    void add_input_layout_constraint() override;

    NodeProp* do_make_node_prop() const override;
};

/*!
 * \brief Merge multiple conditional execution branches, so the output would not
 *      be conditional.
 *
 * Each branch can have multiple vars, and the input array layout is
 * [nr_brahch, nr_output]. Number of outputs is given in the param.
 *
 * The input() is (flattened branch outputs) +
 * [SUM mode only](shape input vars) +
 * [SUM_COND_OUT mode only](predicate var)
 *
 * If mode is SUM_COND_OUT, an extra input from CondExecPredLogical would be
 * added to control the execution of this opr, and it behaves similarly to
 * CondExecMark in such case.
 */
MGB_DEFINE_OPR_CLASS(CondExecMerge, cg::SingleCNOperatorNodeBase) // {
public:
    using ExecutionMask = cg::ExecutionMask;
    using Param = megdnn::param::CondExecMerge;
    using Mode = Param::Mode;

    CondExecMerge(const VarNodeArrayView& inputs,
                  const VarNodeArrayView& out_shapes, const Param& param,
                  const OperatorNodeConfig& config);

    /*!
     * note: \p out_shapes is needed when mode is SUM. If it is nullptr,
     * input shape in at least one branch must be statically inferrable and its
     * shape would be used.
     */
    static OperatorNodeBase* make_opr(const VarNodeArrayView& inputs,
                                      const VarNodeArrayView& out_shapes,
                                      const Param& param,
                                      const OperatorNodeConfig& config);

    static SymbolVarArray make(const VarNodeArrayView& inputs,
                               const Param& param,
                               const VarNodeArrayView& out_shapes = {},
                               const OperatorNodeConfig& config = {}) {
        return cg::to_symbol_var_array(
                make_opr(inputs, out_shapes, param, config)->output());
    }

    const Param& param() const { return m_param; }

    //! input mask of a branch
    ExecutionMask* branch_mask(size_t branch) const {
        return m_branch_masks.at(branch);
    }

    const SmallVector<ExecutionMask*>& branch_masks() const {
        return m_branch_masks;
    }

    /*!
     * \brief merge CondExecMerge oprs for gradient computing
     *
     * This is used by Elemwise::sum_grad_list(). It is necessary because
     * SUM_COND_OUT must be merged or otherwise inputs of gradient add opr may
     * be associated with different ExecutionMask objects.
     */
    static void modify_grad_sum_list(VarNode* wrt, VarNodeArray& grads);

private:
    const Param m_param;

    //! the megdnn opr for SUM/SUM_COND_OUT modes
    intl::UniqPtrWithCN<megdnn::Elemwise> m_exec_dnn_opr;

    std::vector<bool> m_mem_forwarded;

    SmallVector<ExecutionMask*> m_branch_masks;

    void init_output_static_infer_desc() override;
    void add_input_layout_constraint() override;
    void scn_do_execute() override;

    NodeProp* do_make_node_prop() const override;

    bool is_exact_one() const {
        return m_param.mode == Param::Mode::EXACT_ONE ||
               m_param.mode == Param::Mode::EXACT_ONE_SAME_SHAPE;
    }
};

}  // namespace opr
}  // namespace mgb

#endif  // MGB_ENABLE_COND_EXEC

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
