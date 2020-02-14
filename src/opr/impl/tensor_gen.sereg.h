/**
 * \file src/opr/impl/tensor_gen.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/tensor_gen.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace opr {
    MGB_SEREG_OPR(Alloc, 1);
    MGB_SEREG_OPR(Linspace, 3);
    MGB_SEREG_OPR(Eye, 1);
} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

