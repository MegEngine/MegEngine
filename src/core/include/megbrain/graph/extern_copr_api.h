/**
 * \file src/core/include/megbrain/graph/extern_copr_api.h
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

#include "megbrain/graph/bases.h"
#include "megbrain/serialization/extern_c_opr.h"

namespace mgb {

/*!
 * \brief config extern c opr dynamic param
 */
void config_extern_c_opr_dynamic_param(
        std::unique_ptr<cg::AsyncExecutable>& func,
        std::shared_ptr<ExternCOprParam> param);

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
