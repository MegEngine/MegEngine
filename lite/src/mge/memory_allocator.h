/**
 * \file src/mge/memory_alloctor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "lite_build_config.h"

#if LITE_BUILD_WITH_MGE
#include "common.h"
#include "megbrain/dtype.h"
#include "network_impl.h"

#include "megbrain/graph/cg.h"

namespace lite {

class UserStaticMemAlloc final : public mgb::cg::DeviceMemoryAllocator {
    std::shared_ptr<Allocator> m_allocator = nullptr;

public:
    UserStaticMemAlloc(std::shared_ptr<Allocator> allocator)
            : m_allocator(allocator) {}

    void alloc_static(LComputingGraph*, LDeviceTensorStorage& dest,
                      size_t size) override {
        if (size < dest.size()) {
            return;
        }
        auto cn = dest.comp_node_allow_invalid();
        LITE_ASSERT(cn.valid(), "The compnode is invalid when alloc memory.");
        LiteDeviceType device_type =
                get_device_from_locator(cn.locator_logical());
        int device_id = cn.locator_logical().device;
        auto ptr_alloc = static_cast<mgb::dt_byte*>(m_allocator->allocate(
                device_type, device_id, size, cn.get_mem_addr_alignment()));
        auto storage = std::shared_ptr<mgb::dt_byte>(
                ptr_alloc,
                [allocator = m_allocator, device_type, device_id](void* ptr) {
                    allocator->free(device_type, device_id, ptr);
                });
        dest.reset(cn, size, storage);
    }
    void alloc_dynamic(mgb::VarNode*, mgb::DeviceTensorStorage& dest,
                       size_t size) override {
        alloc_static(nullptr, dest, size);
    }

    void defrag_prealloc_contig(mgb::ComputingGraph*, mgb::CompNode comp_node,
                                size_t size) override {
        LiteDeviceType device_type =
                get_device_from_locator(comp_node.locator_logical());
        int device_id = comp_node.locator_logical().device;
        auto ptr_tmp =
                m_allocator->allocate(device_type, device_id, size,
                                      comp_node.get_mem_addr_alignment());
        m_allocator->free(device_type, device_id, ptr_tmp);
    }
};

}  // namespace lite
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
