/**
 * \file imperative/src/include/megbrain/imperative/ops/broadcast.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/tensor_manip.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/op_def.h"

namespace mgb::imperative {

class Broadcast : public OpDefImplBase<Broadcast> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    Broadcast() = default;

    size_t hash() const override {
        return reinterpret_cast<std::uintptr_t>(dyn_typeinfo());
    }

    bool is_same_st(const Hashable& rhs) const override {
        return true;
    }

};

} // namespace mgb::imperative
