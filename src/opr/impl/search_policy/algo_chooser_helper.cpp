/**
 * \file src/opr/impl/search_policy/algo_chooser_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megbrain/opr/search_policy/algo_chooser.h"
#include "megbrain/graph/cg.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace mixin;
/* ==================== misc impl  ==================== */

AlgoChooserHelper::~AlgoChooserHelper() = default;

void AlgoChooserHelper::set_execution_policy(const ExecutionPolicy& policy) {
    mgb_throw_if(
            m_policy_accessed, InternalError,
            "attempt to modify ExecutionPolicy after it has been accessed");
    m_policy = policy;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
