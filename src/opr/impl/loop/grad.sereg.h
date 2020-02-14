/**
 * \file src/opr/impl/loop/grad.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/loop.h"
#include "megbrain/serialization/sereg.h"
#include "./grad_sereg.h"

namespace mgb {
namespace opr {
namespace intl {
    MGB_SEREG_OPR_INTL_CALL_ENTRY(
            LoopGrad, serialization::LoopGradSerializerReg);
} // namespace intl
} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

