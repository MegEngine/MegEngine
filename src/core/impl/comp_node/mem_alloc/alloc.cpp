/**
 * \file src/core/impl/comp_node/mem_alloc/alloc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#include "./impl.h"

#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"

using namespace mgb;
using namespace mem_alloc;

std::unique_ptr<DevMemAlloc> DevMemAlloc::make(
        int device, size_t reserve_size,
        const std::shared_ptr<mem_alloc::RawAllocator>& raw_allocator,
        const std::shared_ptr<mem_alloc::DeviceRuntimePolicy>&
                runtime_policy) {
    mgb_throw_if(!raw_allocator || !runtime_policy, MegBrainError,
                 "raw_alloctor or runtime_policy of device mem allocator is "
                 "not provided, got(raw_allocator:%p, runtime_policy:%p)",
                 raw_allocator.get(), runtime_policy.get());
    return std::make_unique<DevMemAllocImpl>(device, reserve_size,
                                             raw_allocator, runtime_policy);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
