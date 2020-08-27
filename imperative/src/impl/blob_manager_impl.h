/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
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
