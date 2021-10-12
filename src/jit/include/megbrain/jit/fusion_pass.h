/**
 * \file src/jit/include/megbrain/jit/fusion_pass.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/gopt/framework.h"

#if MGB_JIT

namespace mgb {
namespace gopt {

enum class JITFeatureBits : uint32_t {
    NONE = 0,

    //! whether to fuse reduce oprs
    REDUCE = 1,
    //! whether to fuse dimshuffle oprs
    //! DIMSHUFFLE and REDUCE can not coexsit
    DIMSHUFFLE = 2
};

MGB_DEF_ENUM_CLASS_BIT_OPR(JITFeatureBits);

/*!
 * \brief fuse elemwise arith oprs in a subgraph to a fusion opr
 */
class JITFusionPass final : public Pass {
    class Impl;
    bool m_after_grad;
    JITFeatureBits m_feature_bits;

public:
    using JITConfig = cg::ComputingGraph::Options::GraphOpt::JITConfig;

    /*
     * Explanation of how graph_opt_level, jit_opt_level and jit_config
     * control the behavior of JIT optimization:
     *
     * The design of this API is restricted by the historical burden of
     * jit_opt_level and we have to support the old interface jit_opt_level and
     * the new interface jit_config at the same time.
     *
     * How JITFusionPass decides its behavior:
     * (1) When graph_opt_level is 3, it sets jit_opt_level to 1
     * (2) When the user-defined jit_opt_level is greater than 1, it overwrites
     *     the previous value of jit_opt_level
     * (3) We get a default jit_config from jit_opt_level:
     *     jit_opt_level = 0: JIT optimization OFF
     *     jit_opt_level = 1: dimshuffle ON, reduce OFF
     *     jit_opt_level = 2: dimshuffle OFF, reduce ON
     * (4) The user-defined jit_config provides more precise control and
     *     overwrites the default settings defined by jit_opt_level
     *
     * Situations in which JIT optimization is ON:
     * (1) graph_opt_level = 3
     * (2) graph_opt_level = 2, jit_opt_level > 0
     * (3) graph_opt_level = 2, jit_opt_level = 0, jit_config is set
     * (4) graph_opt_level = 0, jit_opt_level > 0 (deprecated usage)
     *
     * Situations in which JIT optimization is OFF:
     * (1) graph_opt_level = 2, jit_opt_level = 0, jit_config is unset
     * (2) graph_opt_level = 1
     * (3) graph_opt_level = 0, jit_opt_level = 0
     */
    JITFusionPass(
            bool after_grad = true, int jit_opt_level = 0,
            const JITConfig& jit_config = {});
    const char* name() const override;
    void apply(OptState& opt) const override;
};

}  // namespace gopt
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
