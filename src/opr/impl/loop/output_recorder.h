/**
 * \file src/opr/impl/loop/output_recorder.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/loop.h"

class mgb::opr::intl::LoopImpl::Desc::OutputRecorderBase: public mgb::Hashable {
    bool is_same_st(const Hashable &) const override {
        return true;
    }

    public:
        using SubgraphStaticInferHelper =
            mgb::cg::static_infer::SubgraphStaticInferHelper;

        virtual ~OutputRecorderBase() = default;

        //! OutputMode for creating this output recorder
        virtual OutputMode output_mode() const = 0;

        size_t hash() const override {
            return mgb::hash(dyn_typeinfo());
        }

        /*!
         * \brief bind var in sub graph to output var of the loop operator
         */
        virtual void bind_var(VarNode *var_sub, VarNode *var_out) = 0;

        /*!
         * \brief whether output var infer desc could be registered
         */
        virtual bool has_shape_infer_desc() const = 0;

        /*!
         * \brief register shape/value infer desc for the output var in parent
         *      graph; called after bind_var() if has_shape_infer_desc() returns
         *      true
         */
        virtual void register_infer_desc(SubgraphStaticInferHelper &) const {
            mgb_assert(0);
        }

        /*!
         * \brief callback before exec begins; note that output var may have not
         *      been allocated here
         */
        virtual void on_exec_begin() {}

        /*!
         * \brief callback on each time the loop body is executed and output var
         *      produced
         */
        virtual void on_val_produced(const DeviceTensorND& val) {
            MGB_MARK_USED_VAR(val);
        }

        /*!
         * \brief callback after loop exits
         */
        virtual void on_exec_end() {}

        /*!
         * \brief name of the output recorder
         */
        virtual std::string name() const = 0;

        /*!
         * \brief get output grad in one iteration in grad
         * \param loop_counter_down counter that goes from loop number downward
         *      to 0
         * \param loop_counter_up counter that goes from 0 to loop number
         */
        virtual SymbolVar get_outgrad_in_iter(
                SymbolVar loop_counter_down, SymbolVar loop_counter_up,
                SymbolVar outgrad) = 0;

};
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

