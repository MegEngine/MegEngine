/**
 * \file src/serialization/include/megbrain/serialization/metadata.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief MegEngine model's metadata
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */
#pragma once

#include <string>

namespace mgb {
namespace serialization {

struct Metadata {
    bool is_valid = false;

    bool graph_modified = false;

    bool has_user_info = false;
    std::string user_info;

    bool optimized_for_inference = false;
    uint64_t optimize_options;

#define ADD_PROPERTY(type, name)       \
    type get_##name() const { return name; } \
    void set_##name(type x) {          \
        name = x;                      \
        has_##name = true;             \
    }
ADD_PROPERTY(std::string, user_info)
#undef ADD_PROPERTY

    uint64_t get_optimize_options() { return optimize_options; }
    void set_optimize_options(uint64_t value) {
        optimized_for_inference = true;
        optimize_options = value;
    }
};

}  // namespace serialization
}  // namespace mgb