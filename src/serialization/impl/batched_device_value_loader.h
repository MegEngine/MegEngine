/**
 * \file src/serialization/impl/batched_device_value_loader.h
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
#include "megbrain/comp_node.h"
#include "megbrain/tensor.h"

namespace mgb {
namespace serialization {

/*!
 * \brief load a batch of DeviceTensorND with a single device allocation and
 *      memory transaction
 *
 * This class caches the host values and merge them and copy them to device in a
 * single transaction. Some devices (like hexagon) have long latency so batching
 * has great benifits.
 */
class BatchedDeviceValueLoader {
    struct TensorList {
        std::vector<std::pair<HostTensorND, std::shared_ptr<DeviceTensorND>>>
                tensors;
    };
    CompNode::UnorderedMap<TensorList> m_cn2tensor_list;

public:
    /*!
     * \brief make a place holder device tensor that has correct dtype and comp
     *      node, but an empty pointer
     * \param comp_node target comp node
     * \param value tensor value; it should be placed on the CPU comp node
     */
    std::shared_ptr<DeviceTensorND> make(CompNode comp_node,
                                         HostTensorND value);

    //! apply all the lazy loads
    void apply();
};

}  // namespace serialization
}  // namespace mgb
