/**
 * \file src/jit/include/megbrain/jit/fusion_pass.h
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
    JITFusionPass(bool after_grad = true, int8_t jit_opt_level = 1);
    const char* name() const override;
    void apply(OptState& opt) const override;
};

}  // namespace gopt
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
