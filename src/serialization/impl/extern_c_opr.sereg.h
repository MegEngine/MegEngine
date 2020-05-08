/**
 * \file src/serialization/impl/extern_c_opr.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {

namespace serialization {
template <>
struct OprLoadDumpImpl<opr::ExternCOprRunner, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        opr::ExternCOprRunner::dump(ctx, opr);
    }

    static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config) {
        return opr::ExternCOprRunner::load(ctx, inputs, config);
    }
};

using ExternCOprRunner = opr::ExternCOprRunner;
MGB_SEREG_OPR(ExternCOprRunner, 0);
MGB_REG_OPR_SHALLOW_COPY(ExternCOprRunner, ExternCOprRunner::shallow_copy);
}  // namespace serialization
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
