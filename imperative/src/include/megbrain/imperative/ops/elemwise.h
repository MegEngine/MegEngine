/**
 * \file imperative/src/include/megbrain/imperative/ops/elemwise.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/basic_arith.h"
#include "megbrain/imperative/op_def.h"

namespace mgb::imperative {

class Elemwise : public OpDefImplBase<Elemwise> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    using Mode = opr::Elemwise::Mode;
    using ModeTrait = megdnn::Elemwise::ModeTrait;

    Mode mode;

    Elemwise() = default;
    
    Elemwise(const Mode& mode_): mode(mode_) {}
    
    size_t hash() const override {
        return hash_pair_combine(mgb::hash(mode), reinterpret_cast<std::uintptr_t>(dyn_typeinfo()));
    }

    bool is_same_st(const Hashable& rhs_) const override {
        auto&& rhs = static_cast<const Elemwise&>(rhs_);
        return rhs.mode == mode;
    }

};

} // namespace mgb::imperative
