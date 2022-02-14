/**
 * \file imperative/src/impl/blob_manager_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/blob_manager.h"

namespace mgb {
namespace imperative {

class BlobManagerImpl final : public BlobManager {
    struct BlobSetWithMux {
        std::mutex mtx;
        ThinHashSet<OwnedBlob*> blobs_set;
        bool insert(OwnedBlob* blob) {
            MGB_LOCK_GUARD(mtx);
            return blobs_set.insert(blob).second;
        }
        size_t erase(OwnedBlob* blob) {
            MGB_LOCK_GUARD(mtx);
            return blobs_set.erase(blob);
        }
    };

    struct BlobData {
        OwnedBlob* blob;
        HostTensorStorage h_storage;
        BlobData(OwnedBlob* in_blob);
    };

    std::mutex m_mtx;
    CompNode::UnorderedMap<BlobSetWithMux> m_comp2blobs_map;

    void defrag(const CompNode& cn) override;

    void alloc_direct(OwnedBlob* blob, size_t size) override;

    DeviceTensorND alloc_workspace(CompNode cn, TensorLayout layout);

    BlobManager::allocator_t custom_allocator;

public:
    static BlobManager* inst();

    void alloc_with_defrag(OwnedBlob* blob, size_t size) override;

    DeviceTensorND alloc_workspace_with_defrag(
            CompNode cn, TensorLayout& layout) override;

    void register_blob(OwnedBlob* blob) override;

    void unregister_blob(OwnedBlob* blob) override;

    void set_allocator(allocator_t allocator) override;
};

}  // namespace imperative
}  // namespace mgb
