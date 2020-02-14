/**
 * \file src/gopt/include/megbrain/gopt/misc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/gopt/framework.h"

namespace mgb {
namespace gopt {

    /*!
     * \brief remove oprs unrelated to computing, such as
     *      MarkNoBroadcastElemwise
     */
    class RemoveNonComputingOprPass final: public Pass {
        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief expand VirtualGrad opr to actual grads
     */
    class ExpandVirtualGradPass final: public Pass {
        public:
            const char* name() const override;
            void apply(OptState &opt) const override;
    };

    /*!
     * \brief delay Broadcast opr after a chain of unary oprs.
     */
    class DelayBroadcastPass final : public Pass {
        static bool allowed_opr(OperatorNodeBase*);

    public:
        const char* name() const override;
        void apply(OptState& opt) const override;
    };

    /*!
     * \brief recompute the TypeCvt if input's dtype_size > output's dtype_size
     *  and long-term dependency exists.
     *      Reduce the memory usage.
     */
    class RecompTypeCvtPass final : public Pass {
    public:
        RecompTypeCvtPass(size_t threshold = 20) : m_threshold(threshold) {}

        const char* name() const override;
        void apply(OptState& opt) const override;

    private:

        //! device whether need to recompute, if the timestamp between two operators exceeding it.
        size_t m_threshold;
    };

    /*!
     * \brief Combine TypeCvt and Reduce operator into a single Reduce opr.
     *      For now, we support 16 -> 32 only.
     */
    class CombineAstypeAndReducePass final : public Pass {
    public:
        const char* name() const override;
        void apply(OptState& opt) const override;
    };

    class RemoveRedundantTypeCvtPass final : public Pass {
    private:
        //! Should we remove the TypeCvt chain of form A -> B -> A?
        static bool should_remove(DType A, DType B);
    public:
        const char * name() const override;
        void apply(OptState &opt) const override;
    };

    //! remove execution mask for const PPVs in conditional execution
    class CondExecConstPredicateFolding final : public Pass {
    public:
        const char* name() const override;
        void apply(OptState& opt) const override;
    };

} // namespace gopt
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
