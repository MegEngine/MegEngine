/**
 * \file imperative/python/src/ops.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./enum_macro.h"
#include "./helper.h"

#include "megbrain/imperative/ops/custom_opdef.h"
#include "megbrain/opr/param_defs.h"
#include "megdnn/opr_param_defs.h"

namespace PYBIND11_NAMESPACE {
namespace detail {

#define ENUM_CASTER_DEF(name)                                           \
    template <>                                                         \
    struct type_caster<name> {                                          \
        PYBIND11_TYPE_CASTER(name, _(#name));                           \
                                                                        \
    public:                                                             \
        bool load(handle src, bool);                                    \
        static handle cast(const name& v, return_value_policy, handle); \
    };

FOR_EACH_ENUM_PARAM(ENUM_CASTER_DEF)
FOR_EACH_BIT_COMBINED_ENUM_PARAM(ENUM_CASTER_DEF)

}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

void init_ops(pybind11::module m);
void init_custom(pybind11::module m);
