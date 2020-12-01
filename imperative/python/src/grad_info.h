/**
 * \file imperative/python/src/grad_info.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <memory>

#include "./intrusive_list.h"

namespace mgb::imperative::python {

struct GradFn;
struct GradSlot;

struct GradSlotPtr {
    std::shared_ptr<GradFn> grad_fn;
    size_t idx;

    GradSlot* operator->();
};

struct GradInfo : GradSlotPtr, intrusive_list::Node<GradInfo, intrusive_list::before_t> {
    GradInfo() = default;
    GradInfo(GradInfo&) = default;
    GradInfo(GradInfo&&) = default;
    GradInfo& operator=(GradInfo&) = default;
    GradInfo& operator=(GradInfo&&) = default;
};

} // namespace mgb::imperative::python
