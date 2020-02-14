/**
 * \file src/opr/impl/loop/forward.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/loop.h"

#include "./forward_sereg.h"

namespace mgb {
namespace opr {
    MGB_SEREG_OPR_INTL_CALL_ENTRY(Loop, serialization::LoopSerializerReg);
    MGB_REG_OPR_SHALLOW_COPY(Loop, serialization::opr_shallow_copy_loop);
} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

