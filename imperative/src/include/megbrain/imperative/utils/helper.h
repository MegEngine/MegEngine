/**
 * \file imperative/src/include/megbrain/imperative/utils/span.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <iomanip>
#include <memory>
#include <sstream>

namespace mgb {

namespace imperative {

template <typename T>
class CleanupGuard {
private:
    T m_callback;

public:
    explicit CleanupGuard(T cb) : m_callback{std::move(cb)} {}
    ~CleanupGuard() { m_callback(); }
};

inline std::string quoted(std::string str) {
    std::stringstream ss;
    ss << std::quoted(str);
    return ss.str();
}

}  // namespace imperative

}  // namespace mgb