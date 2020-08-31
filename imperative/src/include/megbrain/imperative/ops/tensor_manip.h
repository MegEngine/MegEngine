/**
 * \file imperative/src/include/megbrain/imperative/ops/tensor_manip.h
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

class GetVarShape : public OpDefImplBase<GetVarShape> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    GetVarShape() = default;

    size_t hash() const override {
        return reinterpret_cast<std::uintptr_t>(dyn_typeinfo());
    }

    bool is_same_st(const Hashable& rhs) const override {
        return rhs.dyn_typeinfo() == dyn_typeinfo();
    }
};

class ParamPackSplit : public OpDefImplBase<ParamPackSplit> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ParamPackSplit() = default;

    ParamPackSplit(std::vector<dt_int32>& offsets_,
                   std::vector<std::vector<size_t>>& shapes_)
            : offsets(offsets_), shapes(shapes_) {}

    std::vector<dt_int32> offsets;
    std::vector<std::vector<size_t>> shapes;
};

class ParamPackConcat : public OpDefImplBase<ParamPackConcat> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ParamPackConcat() = default;

    ParamPackConcat(std::vector<dt_int32>& offsets_)
            : offsets(offsets_) {}

    std::vector<dt_int32> offsets;
};

} // namespace mgb::imperative
