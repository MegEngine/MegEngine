/**
 * \file src/opr/include/megbrain/opr/search_policy/algo_chooser_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain/graph/operator_node.h"
#include "megbrain/opr/param_defs.h"
#include "megdnn/oprs/base.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace opr {

namespace mixin {

/*!
 * \brief base class for the opr which can be tuning
 */
class AlgoChooserHelper : cg::OperatorNodeMixinBase {
public:
    using ExecutionPolicy = megdnn::param::ExecutionPolicy;
    using AlgorithmInfo = megdnn::detail::Algorithm::Info;
    using AlgoChooserHook =
            std::function<AlgorithmInfo(const cg::OperatorNodeBase*)>;

    const ExecutionPolicy& execution_policy() const {
        if (!m_policy_accessed) {
            m_policy_accessed = true;
        }
        return m_policy;
    }

    /*!
     * \brief get current policy without marking it as having been accessed
     *
     * This is primarily used for getting current policy before calling
     * set_execution_policy().
     */
    const ExecutionPolicy& execution_policy_transient() const {
        return m_policy;
    }

    /*!
     * \brief modify execution policy
     *
     * Exception would be thrown if execution_policy() has been accessed,
     * since it would influence cache and many other decisions.
     */
    void set_execution_policy(const ExecutionPolicy& policy);

    /*!
     * \brief register a hook to implement custom algo chooser
     */
    void setup_algo_chooser(AlgoChooserHook&& func) { m_algo_chooser = func; }
    AlgoChooserHook algo_chooser() const { return m_algo_chooser; }

protected:
    ~AlgoChooserHelper();

    mutable bool m_policy_accessed = false;
    ExecutionPolicy m_policy;

    AlgoChooserHook m_algo_chooser;

};
}  // namespace mixin

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
