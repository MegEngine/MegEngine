/**
 * \file imperative/src/include/megbrain/imperative/grad.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/operator.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative {

struct TangentInfo {
    ValueRef value;
    ValueRef tangent;
};

class TangentTransformation final : public Transformation {
public:
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override;

    ValueRef unwrap(ValueRef value) override { mgb_assert(false); }

    std::string name() const override { return "Tangent"; }
};

}  // namespace mgb::imperative
