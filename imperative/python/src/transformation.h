/**
 * \file imperative/python/src/transformation.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <optional>
#include <string>

#include "pybind11/pybind11.h"

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/value.h"
#include "megbrain/utils/small_vector.h"

namespace mgb::imperative::python {
struct TransformationManager {
    enum Segment {
        ModuleTrace,
        Grad,
        Scalar,
        Trace,
        Eval,
    };

    std::array<std::vector<std::shared_ptr<Transformation>>, 5> segments;

    template <Segment segment>
    void register_at(std::shared_ptr<Transformation> transformation) {
        mgb_assert(segment < segments.size());
        std::shared_ptr<Transformation> next;
        for (size_t i = segment; i < segments.size(); ++i) {
            if (!segments[i].empty()) {
                next = segments[i].back();
                break;
            }
        }
        if (!next) {
            transformation->register_at(Transformation::bottom());
        } else {
            transformation->register_at(next->pos());
        }
        segments[segment].push_back(transformation);
    }

    template <Segment segment>
    void unregister(std::shared_ptr<Transformation> transformation) noexcept {
        mgb_assert(segment < segments.size());
        auto iter = std::find(
                segments[segment].begin(), segments[segment].end(), transformation);
        mgb_assert(iter != segments[segment].end());
        transformation->unregister();
        segments[segment].erase(iter);
    }

    static TransformationManager& get_instance() {
        static TransformationManager sl_instance;
        return sl_instance;
    }
};

class PyValue final
        : public MixinValueImpl<PyValue, ValueKind::Object, pybind11::object> {
public:
    using MixinValueImpl::MixinValueImpl;

    std::string to_string() const {
        return pybind11::str((const pybind11::object&)*this).cast<std::string>();
    }
};

}  // namespace mgb::imperative::python
