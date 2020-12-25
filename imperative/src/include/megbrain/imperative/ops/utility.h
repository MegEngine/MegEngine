/**
 * \file imperative/src/include/megbrain/imperative/ops/utility.h
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

#include <pybind11/pybind11.h>

namespace mgb::imperative {

struct GenericPyOp final : OpDefImplBase<GenericPyOp> {
    pybind11::object obj;

    GenericPyOp(pybind11::object obj_) : obj(std::move(obj_)) {};

    size_t hash() const override {
        return pybind11::hash(obj);
    }

    bool is_same_st(const Hashable& rhs) const override {
        return obj.equal(static_cast<const GenericPyOp&>(rhs).obj);
    }

    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

} // namespace mgb::imperative
