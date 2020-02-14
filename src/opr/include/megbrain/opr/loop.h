/**
 * \file src/opr/include/megbrain/opr/loop.h
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
#include "megbrain/opr/internal/mixin_base.h"

namespace mgb {
namespace opr {

namespace intl {

class LoopGrad;

MGB_DEFINE_CLS_WITH_SUPER(LoopImpl, cg::SingleCNOperatorNodeBase) // {
    public:
        class Desc;

        ~LoopImpl();

        cg::ComputingGraph* get_sub_graph() const;

        //! used for test only, defined in grad.cpp
        static bool& test_check_grad_output_recorder_sum_optimize_success();

    protected:
        class DescImplBase;
        class OutputRecordSpecItem;
        class InputMaker;
        class DepTensorUpdator;
        class FwdDesc;
        class MutableStateSaver;
        class SubgraphDepIter;

        std::unique_ptr<DescImplBase> m_desc;

        std::unique_ptr<MutableStateSaver> m_mutable_state_saver;

        thin_function<size_t()> m_static_loop_time_infer;

        //! number of calls to scn_do_execute(), used for checking fwd and grad
        //! match
        size_t m_nr_scn_do_execute_run = 0;

        void scn_do_execute() override;
        NodeProp* do_make_node_prop() const override;

        //! init m_sub_graph_func from m_desc
        void init_sub_graph_func();

        cg::AsyncExecutable* sub_graph_func() const {
            return m_sub_graph_func.get();
        }

        //! add input vars needed by loop desc
        void add_input_in_desc();

        void add_input_layout_constraint() override;

        /*!
         * note that owner graph in *opr_param* must be to nullptr and would be
         * replaced to `desc->owner_graph()` in the constructor; otherwise
         * accessing owner_graph in the caller while performing a
         * std::move(desc) would result in undefined behaviour because it
         * requires param to be evaluated from left to right
         */
        LoopImpl(const OperatorNodeBaseCtorParam &opr_param,
                std::unique_ptr<DescImplBase> desc);

    private:
        friend class LoopTest;
        friend class LoopSerializer;
        friend class LoopGradSerializer;
        friend class LoopGrad;

        //! for testing: get map from var to whether it is enabled in recorder
        ThinHashMap<VarNode*, bool> test_get_var_rec_spec();

        std::unique_ptr<cg::AsyncExecutable> m_sub_graph_func;
};

} // namesapce intl

/*!
 * \brief loop operator
 *
 * The loop operator maintains its own subgraph; the following happens when it
 * is executed:
 *      1. copy input given by Desc::add_input from original graph
 *      2. execute subgraph
 *      3. update variables given by Desc::assign
 *      4. if loop_condition set by Desc::set_loop_condition evaluates to true
 *          (note that it is evaluated in step 2, before updating vars)
 *          (i.e. non-zero float value, since we do not have dtype for now),
 *         jump to 2
 *      5. copy output given by Desc::add_output to original graph
 */
MGB_DEFINE_OPR_CLASS(Loop, intl::LoopImpl) // {
    public:
        using LoopImpl::Desc;
        using DescMaker = thin_function<void(Desc &desc)>;

        //! extra static params
        struct Param {
            int swap_interval;

            //! number of loop executions between swapping saved mutable states
            //! to host; negative number means to use static inferred value
            //! if possible, or use its absolute value otherwise.
            Param(int swap_interval_ = -5):
                swap_interval{swap_interval_}
            {}
        };

        Loop(std::unique_ptr<FwdDesc> desc, DescMaker desc_maker,
                const Param &param, const OperatorNodeConfig &config);

        /*!
         * \brief create a loop operator with given desc; return value
         *      corresponds to outputs given by Desc::add_output
         * \param desc_maker callback function to construct an operator desc,
         *      which must have no side-effect so a desc could be made for grad
         *      opr
         */
        static SymbolVarArray make(
                DescMaker desc_maker, const Param &param = {},
                const OperatorNodeConfig &config = {});

        /*!
         * a special var used for two purposes:
         *
         * 1. express dependency so loop grad is computed after loop
         * 2. add static infer information for loop times
         *
         * Note that device value of this var is underfined after loop
         * execution.
         */
        VarNode* output_counter_var() const {
            return m_output_counter_var;
        }

        const Param& param() const {
            return m_param;
        }

        static VarNode* grad(
                Loop &opr, size_t wrt_idx, const VarNodeArray &out_grad);
    private:
        /*!
         * for static infer of final counter value; the final loop counter value
         * (i.e. the value when loop exits; the value when loop condition is
         * false for the first time) is value for the var passed to the functor.
         * This is setup by do_make_node_prop()
         */
        mutable std::pair<VarNode*,
                thin_function<size_t(const DeviceTensorND &)>>
                    m_static_final_counter_value_infer = {nullptr, {}};

        const Param m_param;
        DescMaker m_desc_maker;
        ThinHashMap<VarNode*, intl::LoopGrad*> m_loss2grad_opr;
        VarNode* m_output_counter_var = nullptr;

        void init_output_static_infer_desc() override;

        /*!
         * \brief create a MutableStateSaver, record impure oprs, and store it
         *      in m_mutable_state_saver
         */
        void init_mutable_state_saver();

        NodeProp* do_make_node_prop() const override;

        void add_input_layout_constraint() override;

        static void optimize_fwd_graph(int level, FwdDesc& desc);
};

namespace intl {
/*!
 * \brief loop operator descriptor
 */
class LoopImpl::Desc: public NonCopyableObj {
    public:
        /*!
         * \brief base class for output recorders, which are used to record the
         *      output values
         */
        class OutputRecorderBase;

        //! output mode enums; deterministic size for serizalization
        enum class OutputMode: uint8_t {
            LAST,   //!< only record the last value of corresponding var
            ALL,    //!< record all values and concat them on the first dim
            SUM,    //!< record sum of the values
        };

        /*!
         * \brief add an input in the loop subgraph, connecting to *inp* in
         *      original graph
         *
         * \param has_assign whether this input var would be assigned later
         */
        virtual SymbolVar add_input(SymbolVar inp, bool has_assign = false) = 0;

        //! helper for add_input(., true)
        SymbolVar add_input_assignable(SymbolVar inp) {
            return add_input(inp, true);
        }

        /*!
         * \brief assign value *val* to *dest* at end of loop, where *dest* must
         *      be a var returned by add_input, and *val* is a var in the
         *      subgraph
         *
         * If the same *dest* is provided multiple times, *val* given at the
         * last time would be used
         */
        virtual Desc& assign(SymbolVar dest, SymbolVar val) = 0;

        /*!
         * \brief mark a symbol as output so it could be recorded and merged
         *      into the output of this loop operator
         * \return id of the output var (i.e. its index in the output vector)
         */
        virtual size_t add_output(SymbolVar val, OutputMode mode);

        /*!
         * \brief get the counter var, which contains number of loops already
         *      completed so far
         */
        virtual SymbolVar get_counter_var() = 0;

        /*!
         * \brief continue loop when *cond* evaluates to true; this is like a
         *      do-while loop in C/C++
         */
        virtual Desc& set_loop_condition(SymbolVar cond) = 0;

        virtual ~Desc() = default;

    protected:

        /*!
         * \brief implement add_output() by subclasses; output mode has been
         *      translated to OutputRecorderBase
         */
        virtual size_t do_add_output(SymbolVar val,
                std::unique_ptr<OutputRecorderBase> recorder) = 0;
};

} // namesapce intl

} // opr
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

