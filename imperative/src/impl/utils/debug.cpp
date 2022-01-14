/**
 * \file imperative/src/impl/utils/debug.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <typeindex>

#include "megbrain/imperative/transformation.h"
#include "megbrain/imperative/utils/debug.h"
#include "megbrain/imperative/value.h"

namespace mgb::imperative::debug {

const char* get_type_name(const std::type_info& type) {
    return type.name();
}

const char* get_type_name(const std::type_index& type) {
    return type.name();
}

void notify_event(const char* event) {}

void watch_value(ValueRef value) {
    value.watch();
}

}  // namespace mgb::imperative::debug