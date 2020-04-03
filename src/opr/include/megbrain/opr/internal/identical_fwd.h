/**
 * \file src/opr/include/megbrain/opr/internal/identical_fwd.h
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


namespace mixin {

//! for internal use by DynamicOutputIfInputDynamic
void init_rt_force_dynamic_mem_alloc_imply_chain_for_dyn_pass_i2o(
        OperatorNodeBase &opr);

/*!
 * \brief mixin for operators which essentially works by forwarding subtensor of
 *      input(0) to output(0) in a readonly way
 */
class ReadonlyFwdHelper: public cg::OperatorNodeMixinBase {
    bool m_mem_fwd_success = false;

    protected:
        SubTensorSpec m_rofwd_subspec;

        /*!
         * \brief call this function in mem_plan_fwd_in2out_readonly() after
         *      m_rofwd_subspec has been setup
         */
        void mixin_rofwd_init_mem_plan(OperatorNodeBase &opr);

        /*!
         * \brief call this function in do_execute() if
         *      mixin_rofwd_init_mem_plan() has been called during mem
         *      optimation
         */
        void mixin_rofwd_execute(OperatorNodeBase &opr);
};

/*!
 * \brief base class for operators whose output value should be the same as its
 *      input value
 *
 * Note: this opr is usually used to introduce side effects into the graph, so
 * the output value is statically inferrable but would never be a const.
 */
class ForwardInputToOutput: public cg::OperatorNodeMixinBase {
    bool m_mem_fwd_success = false, m_ignore_side_effect = false,
            m_static_infer_called = false;

    class MutableSrc;
    protected:
        bool m_append_one_more_shape = false;
        ~ForwardInputToOutput() = default;

        using ValueInferFunc = cg::static_infer::ValueInferDesc::infer_func_t;

        virtual void mixin_scn_do_execute(OperatorNodeBase &opr);

        void mixin_mem_plan_fwd_in2out_readonly(OperatorNodeBase &opr);
        void mixin_init_output_static_infer_desc(OperatorNodeBase &opr);
        virtual cg::static_infer::ValueInferDesc mixin_get_static_infer_desc(OperatorNodeBase &opr);

        //! overwritten by subclass to be notified at the end of scn_do_execute
        virtual void scn_do_execute_finish(const DeviceTensorND &val);

        /*!
         * \brief Set that this opr could ignore side effect.
         *
         * Setting this option allows the static value inference on output var
         * to be constant if the input is constant.
         *
         * Without setting this option, the output would never be constant, so
         * this opr would not be optimized out.
         *
         * This method should be called from the constructor.
         */
        void set_ignore_side_effect();

        /*!
         * \brief register stream propagate function which forwards the
         * StreamPropType from \p opr input(0) to output(0).
         */
        void register_stream_propagate_in2out(OperatorNodeBase &opr);

    public:

        /*!
         * \brief add a mutable dep entry if the desc dep is constant
         *
         * This ensures that the output var would not be optimized by const
         * folding. An extra mutable shape dependency would be appended if all
         * current dependencies are constant.
         *
         * \param[in,out] desc current value inference
         * \return bool, return true if an extra mutable shape dependency be
         * appended (also means all current dependencies are constant.)
         */
        static bool ensure_not_replaced_by_const_folding(
                cg::static_infer::ValueInferDesc& desc);
};

} // namespace mixin

namespace intl {

/*!
 * \brief setup rt_force_dynamic_mem_alloc_imply_chain between input/output
 *
 * There must be exactly output var which has no VOLATILE_CONTENT flag; that
 * output would be added to imply chain of all inputs, and input(0) would be
 * added to the imply chain of that output.
 *
 * Used for two purposes:
 *
 *  1. readonly forwarding in cases with dynamic input and static output shape;
 *  2. ensure mem_plan_fwd_in2out_readonly would always be called to initialize
 *     internal states
 */
template<class Base>
class DynamicOutputIfInputDynamic: public mixin::CheckBase<Base>::Base {
    protected:
        using Base::Base;
        void init_rt_force_dynamic_mem_alloc_imply_chain() override {
            mixin::init_rt_force_dynamic_mem_alloc_imply_chain_for_dyn_pass_i2o(
                    *this);
        }
};

/*!
 * \brief glue class for apply ReadonlyFwdHelper mixin
 *
 * Note that DynamicOutputIfInputDynamic is implicitly added by this helper
 */
template<class Base, class MixinImpl = mixin::ReadonlyFwdHelper>
MGB_DEFINE_CLS_WITH_SUPER(ReadonlyFwdHelper,
        DynamicOutputIfInputDynamic<typename mixin::CheckBase<Base>::Base>,
        public MixinImpl) // {

    protected:
        using Super::Super;

        void rofwd_init_mem_plan() {
            this->mixin_rofwd_init_mem_plan(*this);
        }

        void rofwd_execute() {
            this->mixin_rofwd_execute(*this);
        }
};

/*!
 * \brief base class (already includes OperatorNodeBase) for i2o oprs
 */
MGB_DEFINE_CLS_WITH_SUPER(ForwardInputToOutput,
        cg::SingleCNOperatorNodeBase,
        public mixin::ForwardInputToOutput) // {

    void scn_do_execute() override final {
        mixin_scn_do_execute(*this);
    }

    protected:
        using Super::Super;
        void init_rt_force_dynamic_mem_alloc_imply_chain() override {
            mixin::init_rt_force_dynamic_mem_alloc_imply_chain_for_dyn_pass_i2o(
                    *this);
        }

        void mem_plan_fwd_in2out_readonly() override {
            this->mixin_mem_plan_fwd_in2out_readonly(*this);
        }

        void init_output_static_infer_desc() override {
            this->mixin_init_output_static_infer_desc(*this);
        }

        void init_output_comp_node() override {
            Super::init_output_comp_node();
            this->register_stream_propagate_in2out(*this);
        }
};

} // namespace intl
} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
