/**
 * \file src/core/include/megbrain/graph/grad_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/cg.h"

namespace mgb {
namespace cg {
    //! result of opr grad func; see docs of OprGradFunc
    class OprGradResult {
        bool m_from_single = false;
        VarNode* m_single = nullptr;
        VarNodeArray m_all;

    public:
        OprGradResult() = default;
        OprGradResult(VarNode* var) : m_from_single{true}, m_single{var} {}
        OprGradResult(VarNodeArray arr)
                : m_from_single{false}, m_all{std::move(arr)} {}

        bool from_single() { return m_from_single; }
        VarNode* single() { return m_single; }

        //! check that m_all.size() matches opr->input().size(), and return
        //! m_all
        VarNodeArray& all(OperatorNodeBase* opr);
    };

    /*!
     * \brief Compute the grad of a scalar with respect to one of the
     *      inputs of an operator; return nullptr if result does not depend on
     *      the input;
     *
     * wrt_idx and out_grad are guaranteed to be valid. Note that if target does
     * not depend on output(i), out_grad[i] would be nullptr; it is guaranteed
     * that at least one element in out_grad is not nullptr.
     *
     * Note: some oprs may benefit from computing the gradients of all inputs at
     * once. In such case, it can return the gradients of all inputs as an
     * VarNodeArray, and the opr grad func would not be invoked again for the
     * same set of output grads.
     *
     * IMPORTANT: the grad func should only access input/output vars by
     * opr->input() or opr->output() (rather than by copied member vars, which
     * should not exist in the first place anyway), since input/output vars may
     * be replaced for reusing a grad func (e.g. see call_opr_grad_on_given_io).
     */
    using OprGradFunc = thin_function<OprGradResult(
            OperatorNodeBase *opr,
            size_t wrt_idx, const VarNodeArray &out_grad)>;

    //! a callback that could modify grad var; see add_grad_transformer()
    using GradTransformer = thin_function<VarNode*(
            VarNode *target, VarNode *wrt, VarNode *grad)>;

    //! a callback that acts like OprGradFunc to compute grad of input vars;
    //! see add_var_virtual_receiver()
    using VarVirtualReceiverGrad = thin_function<VarNode*(
            const VarNodeArray &inputs, const VarNodeArray &outputs,
            size_t wrt_idx, const VarNodeArray &out_grad)>;

    /*!
     * \brief register grad func for an operator type
     */
    void register_grad_func(Typeinfo *opr_type, OprGradFunc grad);

    /*!
     * \brief add a callback to be invoked when grad of given var is computed
     *
     * All transformers would be chained in their added order, and the last
     * return value would be used as grad var.
     *
     * Remember to call add_extra_dep_for_grad if the GradTransformer needs to
     * compute grad on other var.
     */
    void add_grad_transformer(VarNode *var, const GradTransformer &cb);

    /*!
     * \brief set a callback to compute the gradient of *inputs*
     *
     * The given callback would be treated like an operator that receives
     * *inputs* and produces *outputs*, to compute the gradient of *inputs*.
     *
     * Note: graph transformation should be disabled until grad has been
     * computed if virtual receiver is needed
     */
    void add_var_virtual_receiver(
            const VarNodeArray &inputs, const VarNodeArray &outputs,
            const VarVirtualReceiverGrad &grad);

    /*!
     * \brief reuse grad func registered by an operator to implement grads
     *      between given inputs and outputs
     *
     * This is implemented by add_var_virtual_receiver
     *
     * \param add_volatile_out see call_opr_grad_on_given_io
     */
    void add_var_virtual_receiver_reuse_opr_grad(
            const VarNodeArray &inputs, const VarNodeArray &outputs,
            OperatorNodeBase *opr, bool add_volatile_out);

    /*!
     * \brief add an edge in the dependency graph
     *
     * This function claims that \p out depends on \p inp in forward computing
     * graph, so when computing gradients, \p inp would be considered to
     * contribute to target var if \p out contributes to target var.
     */
    void add_extra_dep_for_grad(VarNode *inp, VarNode *out);

    /*!
     * \brief call registered OprGradFunc on given input and output vars
     *
     * This helper is useful to implement grad in output var replacing (e.g.
     * used in Loop)
     *
     * \param add_volatile_out whether to add null vars in the place of volatile
     *      output vars to outputs
     */
    VarNode* call_opr_grad_on_given_io(
            OperatorNodeBase *opr,
            const VarNodeArray &inputs, const VarNodeArray &outputs,
            size_t idx, const VarNodeArray &out_grad,
            bool add_volatile_out);

    //! helper class to call register_grad_func() in the constructor
    class OprGradRegCaller {
        public:
            template<class ...Args>
            OprGradRegCaller(Args&&...args) {
                register_grad_func(std::forward<Args>(args)...);
            }
    };

#if MGB_ENABLE_GRAD
#define MGB_REGISTER_GRAD_FUNC(_opr_type, _func) \
    namespace { \
        ::mgb::cg::OprGradRegCaller _reg_grad_##_opr_type( \
                _opr_type::typeinfo(), _func); \
    }
#else
#define MGB_REGISTER_GRAD_FUNC(_opr_type, _func)
#endif

/*!
 * \brief helper macro for implementing operator grad func
 *
 * This macro would start declaring a function, so it should be followed by a
 * pair of braces which define the function body; the function signature would
 * be (const _opr_type &opr, size_t wrt_idx, const VarNodeArray &out_grad).
 */
#define MGB_IMPL_OPR_GRAD(_opr_type)                                      \
    namespace {                                                           \
    struct _OprGradImpl##_opr_type {                                      \
        static ::mgb::cg::OprGradResult impl(                             \
                const _opr_type& opr, size_t wrt_idx,                     \
                const ::mgb::cg::VarNodeArray& out_grad);                 \
        static ::mgb::cg::OprGradResult wrap(                             \
                ::mgb::cg::OperatorNodeBase* opr, size_t wrt_idx,         \
                const ::mgb::cg::VarNodeArray& out_grad) {                \
            return impl(opr->cast_final<_opr_type>(), wrt_idx, out_grad); \
        }                                                                 \
    };                                                                    \
    }                                                                     \
    MGB_REGISTER_GRAD_FUNC(_opr_type, _OprGradImpl##_opr_type::wrap);     \
    ::mgb::cg::OprGradResult _OprGradImpl##_opr_type::impl(               \
            const _opr_type& opr, size_t wrt_idx,                         \
            const ::mgb::cg::VarNodeArray& out_grad)

} // cg
} //mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
