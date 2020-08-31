/**
 * \file imperative/src/include/megbrain/imperative/ops/cond_take.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb::imperative {

class CondTake : public OpDefImplBase<CondTake> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    CondTake() = default;
};

} // namespace mgb::imperative
