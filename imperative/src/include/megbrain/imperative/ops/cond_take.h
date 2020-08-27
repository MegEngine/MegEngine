/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
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
