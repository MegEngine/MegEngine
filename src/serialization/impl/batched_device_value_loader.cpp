/**
 * \file src/serialization/impl/batched_device_value_loader.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "batched_device_value_loader.h"

#include "megbrain/utils/arith_helper.h"

namespace mgb {
namespace serialization {

std::shared_ptr<DeviceTensorND> BatchedDeviceValueLoader::make(
        CompNode comp_node, HostTensorND value) {
    auto&& tensor_list = m_cn2tensor_list[comp_node];
    auto dev_tensor = std::make_shared<DeviceTensorND>();
    DeviceTensorStorage storage;

    auto size = value.layout().span().dist_byte();
    storage.reset(comp_node, size, nullptr);
    dev_tensor->reset(storage, value.layout());
    tensor_list.tensors.emplace_back(std::move(value), dev_tensor);
    return dev_tensor;
}

void BatchedDeviceValueLoader::apply() {
    for (auto&& item : m_cn2tensor_list) {
        auto alignment = item.first.get_mem_addr_alignment();
        size_t tot_size = 0;
        for (auto&& i : item.second.tensors) {
            tot_size = get_aligned_power2(tot_size, alignment) +
                       i.second->layout().span().dist_byte();
        }

        HostTensorStorage host_storage{item.first};
        DeviceTensorStorage dev_storage{item.first};
        host_storage.ensure_size(tot_size);
        dev_storage.ensure_size(tot_size);
        auto ptr_host = host_storage.ptr();
        size_t offset = 0;
        for (auto&& i : item.second.tensors) {
            offset = get_aligned_power2(offset, alignment);
            auto size = i.second->layout().span().dist_byte();
            if (i.second->layout().format.is_default()) {
                mgb_assert(size == i.first.layout().span().dist_byte());
                memcpy(ptr_host + offset, i.first.raw_ptr(), size);
            } else {
                HostTensorND host;
                host.reset(host_storage.sub(offset), i.second->layout());
                host.copy_from_fixlayout(i.first);
            }
            i.second->reset(dev_storage.sub(offset), i.second->layout());
            offset += size;
        }
        dev_storage.copy_from(host_storage, tot_size);
        item.first.sync();
    }
}

}  // namespace serialization
}  // namespace mgb
