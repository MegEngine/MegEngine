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

#include "megbrain/utils/hash.h"

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

    size_t hash() const override {
        XXHash builder;
        for (auto&& offset : offsets) {
            builder.update(&offset, sizeof(offset));
        }
        auto&& offset_cnt = offsets.size();
        builder.update(&offset_cnt, sizeof(offset_cnt));
        for (auto&& shape : shapes) {
            for (auto&& dim_len : shape) {
                builder.update(&dim_len, sizeof(dim_len));
            }
            auto&& dim_cnt = shape.size();
            builder.update(&dim_cnt, sizeof(dim_cnt));
        }
        auto&& shape_cnt = shapes.size();
        builder.update(&shape_cnt, sizeof(shape_cnt));
        return builder.digest();
    }

    bool is_same_st(const Hashable& rhs) const override {
        auto* pps = rhs.try_cast_final<ParamPackSplit>();
        if(pps == nullptr){
            return false;
        }
        return offsets == pps->offsets && shapes == pps->shapes;
    }
};

class ParamPackConcat : public OpDefImplBase<ParamPackConcat> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    ParamPackConcat() = default;

    ParamPackConcat(std::vector<dt_int32>& offsets_)
            : offsets(offsets_) {}

    std::vector<dt_int32> offsets;

    size_t hash() const override {
        XXHash builder;
        for (auto&& offset : offsets) {
            builder.update(&offset, sizeof(offset));
        }
        auto&& offset_cnt = offsets.size();
        builder.update(&offset_cnt, sizeof(offset_cnt));
        return builder.digest();
    }

    bool is_same_st(const Hashable& rhs) const override {
        auto* ppc = rhs.try_cast_final<ParamPackConcat>();
        if(ppc == nullptr){
            return false;
        }
        return offsets == ppc->offsets;
    }
};

} // namespace mgb::imperative
