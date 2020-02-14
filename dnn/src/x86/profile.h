/**
 * \file dnn/src/x86/profile.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <vector>

namespace megdnn {
namespace x86 {

struct ProfileElement {
    // when output_size > on_threshold, DIRECT is faster,
    // otherwise MATRIX_MUL is faster
    int f, ic, oc, on_threshold;
    ProfileElement(int f, int ic, int oc, int on_threshold):
        f(f), ic(ic), oc(oc), on_threshold(on_threshold)
    {
    }
    bool operator<(const ProfileElement &rhs) const
    {
        if (this->f < rhs.f) return true;
        if (this->f > rhs.f) return false;
        if (this->ic < rhs.ic) return true;
        if (this->ic > rhs.ic) return false;
        if (this->oc < rhs.oc) return true;
        if (this->oc > rhs.oc) return false;
        return false;
    }
};
using ProfileCache = std::vector<ProfileElement>;

ProfileCache get_profile_cache();

} // namespace fallback
} // namespace megdnn

// vim: syntax=cpp.doxygen


