/**
 * \file src/opr/impl/loop/grad_sereg.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./grad_sereg.h"
#include "./grad.h"
#include "./impl.h"
#include "megbrain/serialization/sereg.h"
#include "megbrain/opr/internal/param_tag_defs.h"

using namespace mgb;
using namespace mgb::serialization;
using namespace mgb::opr::intl;

namespace mgb {
namespace opr {
namespace intl {

//! this is a friend class of LoopImpl and LoopGrad
class LoopGradSerializer {
    template<class Opr>
    static cg::OperatorNodeBase* wrap_shallow_copy(
            const OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr, const VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        MGB_MARK_USED_VAR(ctx);
        return opr.cast_final_safe<Opr>().shallow_copy(inputs, config);
    }
    public:
        static void reg_all();
};

} // namespace intl
} // namespace opr
} // namespace mgb

void LoopGradSerializer::reg_all() {
#define REG(_opr) \
    MGB_REG_OPR_SHALLOW_COPY_IMPL(_opr, wrap_shallow_copy<_opr>)

    REG(LoopGrad);
    REG(LoopGrad::AssignorGradOpr);
    REG(LoopImpl::DepTensorUpdator);

#undef REG
}

void LoopGradSerializerReg::entry() {
    LoopGradSerializer::reg_all();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

