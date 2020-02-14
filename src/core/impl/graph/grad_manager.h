/**
 * \file src/core/impl/graph/grad_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./impl_common.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/utils/mempool.h"

namespace mgb {
namespace cg {

#if MGB_ENABLE_GRAD

/*!
 * \brief compute symbolic gradients
 */
class GradManager {
    public:
        struct VarVirtualReceiverDesc {
            VarNodeArray inputs, outputs;
            VarVirtualReceiverGrad grad;
        };

        GradManager(ComputingGraphImpl *graph);
        ~GradManager() noexcept;

        VarNode* grad(VarNode* target, VarNode *wrt);

        VarNode* current_grad_target() const {
            return m_target_stack.empty() ? nullptr : m_target_stack.back();
        }

        void add_grad_transformer(VarNode *var, const GradTransformer &cb) {
            m_grad_transformers[var].emplace_back(cb);
        }

        void add_extra_dep_for_grad(VarNode *inp, VarNode *out) {
            m_extra_deps_inv_lookup[out].push_back(inp);
        }

        void add_var_virtual_receiver(
                const std::shared_ptr<VarVirtualReceiverDesc> &desc);

        void clean_cache() {
            for (auto &&i : m_target_context) {
                i.second.cache.clear();
                i.second.holistic_input_grads.clear();
            }
        }

    private:
        using VarMap = ThinHashMap<VarNode*, VarNode*>;

        //! whether a var's grad should be put in copy stream with
        //! SeqCompNodeOptimizer::StreamPropType::STRONG
        class StreamStrongPropInfer;

        //! context for grad computing for the same target var
        class ContextForTargetVar {
            size_t m_virtual_receiver_version = 0;
            //! oprs that this target var dependent on
            ThinHashSet<OperatorNodeBase*> m_dep_oprs;

            public:
                //! wrt -> grad
                VarMap cache;

                //! cache for oprs that return grads of all inputs at once
                ThinHashMap<OperatorNodeBase*, VarNodeArray>
                    holistic_input_grads;

                bool has_dep_opr(OperatorNodeBase *opr) const {
                    return m_dep_oprs.count(opr);
                }

                void init(GradManager *manager, VarNode *target);
        };

        using VarVirtualReceiverArray = std::vector<
            std::shared_ptr<VarVirtualReceiverDesc>>;

        //! a single receiver of a var, either an opr or a virtual receiver
        struct VarReceiver {
            OperatorNodeBase * const opr = nullptr;
            VarVirtualReceiverDesc * const vrt = nullptr;

            VarReceiver() = default;
            VarReceiver(OperatorNodeBase *o): opr{o} {}
            VarReceiver(VarVirtualReceiverDesc *v): vrt{v} {}
        };
        using VarReceiverArray = std::vector<VarReceiver>;

        ComputingGraphImpl * const m_owner_graph;

        std::unique_ptr<StreamStrongPropInfer>
            m_stream_strong_prop_infer;
        ThinHashMap<VarNode*, ContextForTargetVar> m_target_context;

        //! current (target, wrt) pairs that are being computed, to detect
        //! infinite recurision
        std::unordered_set<std::pair<VarNode*, VarNode*>, pairhash> m_in_stack;

        //! var -> cb, so when grad of var is computed, cb should be applied
        ThinHashMap<VarNode*, std::vector<GradTransformer>>
            m_grad_transformers;

        //! a -> b, that b depends on a in forward graph
        ThinHashMap<VarNode*, VarNodeArray> m_extra_deps_inv_lookup;

        size_t m_virtual_receiver_version = 0;
        //! var -> corresponding virtual receivers
        ThinHashMap<VarNode*, VarVirtualReceiverArray>
            m_var2virtual_receiver;

        //! var -> virtual receivers that have the var as one of its outputs
        ThinHashMap<VarNode*, std::vector<VarVirtualReceiverDesc*>>
            m_var2virtual_receiver_inv;

        //! stack of grad target vars
        std::vector<VarNode*> m_target_stack;

        //! list of (var, readers)
        using DepSeq = std::vector<std::pair<VarNode*, VarReceiverArray>>;
        struct GetDepSeqStackFrame;

        /*!
         * \brief get all vars that depend on *start_var* in reverse topological
         *      order
         *
         * This method returns a sequence, such that all needed information for
         * computing grad for any var in the sequence is available if grads for
         * all preceding vars have been computed.
         *
         * \return dep sequence, where each element contains a var and its reader
         *      oprs (i.e. those that have that var as input)
         */
        DepSeq get_dep_seq(VarNode *start_var,
                const ContextForTargetVar &tgt_context);

        VarNode* do_grad_with_cache(VarNode* target, VarNode *wrt);

        VarNode* compute_grad_of_single_var(
                VarNode* target, VarNode *wrt,
                ContextForTargetVar& context,
                const VarReceiverArray &wrt_recv,
                VarNodeArray *tmp_var_arrs);
};

#else

class GradManager {
public:
    GradManager(ComputingGraphImpl *) {}
};

#endif  // MGB_ENABLE_GRAD

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}


