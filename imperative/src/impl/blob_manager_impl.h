/**
 * \file imperative/src/impl/blob_manager_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/blob_manager.h"

namespace mgb {
namespace imperative {

class BlobManagerImpl final: public BlobManager {

    struct BlobSetWithMux {
        std::mutex mtx;
        ThinHashSet<Blob*> blobs_set;
        bool insert(Blob* blob) {
            MGB_LOCK_GUARD(mtx);
            return blobs_set.insert(blob).second;
        }
        size_t erase(Blob* blob) {
            MGB_LOCK_GUARD(mtx);
            return blobs_set.erase(blob);
        }
    };

    struct BlobData {
        Blob* blob;
        HostTensorStorage h_storage;
        BlobData(Blob* in_blob);
    };

    std::mutex m_mtx;
    CompNode::UnorderedMap<BlobSetWithMux> m_comp2blobs_map;
    bool m_enable;

    void defrag(const CompNode& cn) override;

    void alloc_direct(Blob* blob, size_t size);

public:
    static BlobManager* inst();

    void alloc_with_defrag(Blob* blob, size_t size) override;

    void register_blob(Blob* blob) override;

    void unregister_blob(Blob* blob) override;

    void set_enable(bool flag) override;
};

} // namespace imperative
} // namespace mgb
