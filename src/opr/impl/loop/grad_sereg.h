/**
 * \file src/opr/impl/loop/grad_sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./grad.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
    struct LoopGradSerializerReg {
        //! entry for registering serializers related to loop grad
        static void entry();
    };

    cg::OperatorNodeBase* opr_shallow_copy_loop_grad(
            const OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr, const VarNodeArray &inputs,
            const OperatorNodeConfig &config);
} // namespace serialization
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

