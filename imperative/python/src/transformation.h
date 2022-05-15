#pragma once

#include <optional>
#include <string>

#include "pybind11/pybind11.h"

#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/value.h"
#include "megbrain/utils/small_vector.h"

namespace mgb::imperative::python {
struct TransformationManager {
public:
    enum Segment {
        ModuleTrace,
        DTypePromote,
        DimExpansion,
        Grad,
        Scalar,
        Symbol,
        Trace,
        Eval,
    };

    std::array<std::vector<std::shared_ptr<Transformation>>, 8> segments;

private:
    template <Segment segment>
    void unregister(std::shared_ptr<Transformation> transformation) noexcept {
        mgb_assert(segment < segments.size());
        auto iter = std::find(
                segments[segment].begin(), segments[segment].end(), transformation);
        mgb_assert(iter != segments[segment].end());
        transformation->unregister();
        segments[segment].erase(iter);
    }

public:
    template <Segment segment>
    [[nodiscard]] std::unique_ptr<CleanupGuard<>> register_at(
            std::shared_ptr<Transformation> transformation) {
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
        return std::make_unique<CleanupGuard<>>(
                [this, transformation]() { unregister<segment>(transformation); });
    }

    static TransformationManager& get_instance() {
        static TransformationManager sl_instance;
        return sl_instance;
    }
};

class PyValue final : public PrimitiveValue<PyValue, pybind11::object> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const {
        return pybind11::str((const pybind11::object&)*this).cast<std::string>();
    }
};

}  // namespace mgb::imperative::python
