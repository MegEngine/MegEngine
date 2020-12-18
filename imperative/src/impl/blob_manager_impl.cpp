/**
 * \file imperative/src/impl/blob_manager_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./blob_manager_impl.h"
#include "megbrain/utils/arith_helper.h"
#include <set>

namespace mgb {
namespace imperative {

BlobManagerImpl::BlobData::BlobData(Blob* in_blob){
    blob = in_blob;
    DeviceTensorStorage d_storage;
    d_storage.reset(blob->m_comp_node, blob->m_size, blob->m_storage);

    h_storage = HostTensorStorage(blob->m_comp_node);

    h_storage.ensure_size(blob->m_size);

    h_storage.copy_from(const_cast<DeviceTensorStorage&>(d_storage), blob->m_size);
}

void BlobManagerImpl::register_blob(Blob* blob) {
    // add blob into the comp2blobs map
    MGB_LOCK_GUARD(m_mtx);
    mgb_assert(m_comp2blobs_map[blob->m_comp_node].insert(blob));
}

void BlobManagerImpl::unregister_blob(Blob* blob) {
    // erase blob into the comp2blobs map
    MGB_LOCK_GUARD(m_mtx);
    mgb_assert(1 == m_comp2blobs_map[blob->m_comp_node].erase(blob));
}

void BlobManagerImpl::alloc_with_defrag(Blob* blob, size_t size) {
    if (!m_enable) {
        alloc_direct(blob, size);
    } else {
        // // debug
        // defrag(blob->m_comp_node);
        // alloc_direct(blob, storage, size);

        // try alloc
        MGB_TRY { alloc_direct(blob, size); }
        // if fail, try defrag, alloc again
        MGB_CATCH(MemAllocError&, {
            mgb_log_warn("memory allocation failed for blob; try defragmenting");
            defrag(blob->m_comp_node);
            alloc_direct(blob, size);
        });
    }
}


void BlobManagerImpl::alloc_direct(Blob* blob, size_t size) {
    DeviceTensorStorage storage(blob->m_comp_node);
    mgb_assert(blob->m_comp_node.valid());
    storage.ensure_size(size);
    blob->m_storage = storage.raw_storage();
}

void BlobManagerImpl::defrag(const CompNode& cn) {
    BlobSetWithMux* blobs_set_ptr;
    {
        MGB_LOCK_GUARD(m_mtx);
        blobs_set_ptr = &m_comp2blobs_map[cn];
    }
    MGB_LOCK_GUARD(blobs_set_ptr->mtx);
    std::vector<BlobData> blob_data_arrary;
    std::set<Blob::RawStorage> storage_set;

    auto alignment = cn.get_mem_addr_alignment();
    size_t tot_sz = 0;

    // copy to HostTensorStorage, and release
    for (auto i : blobs_set_ptr->blobs_set) {
        // skip if blob do not have m_storage
        if (!i->m_storage) continue;

        // skip if ues_count() > 1
        if (i->m_storage.use_count() > 1) continue;

        // two blobs can't share same storage
        mgb_assert(storage_set.insert(i->m_storage).second);

        tot_sz += get_aligned_power2(i -> m_size, alignment);
        BlobData blob_data(i);
        blob_data_arrary.push_back(blob_data);
        i -> m_storage.reset();
    }
    // clear all, make sure m_storage will be release
    storage_set.clear();

    // skip if no blob to defrag
    if (!blob_data_arrary.size()) return;

    // wait all other comp nodes to avoid moved var being read; note that
    // ExecEnv has been paused, so no new task would not be dispatched
    CompNode::sync_all();
    CompNode::try_coalesce_all_free_memory();

    // try free all
    MGB_TRY{cn.free_device(cn.alloc_device(tot_sz));}
    MGB_CATCH(MemAllocError&, {})

    // sort blobs by created time, may be helpful for reduce memory fragment
    std::sort(blob_data_arrary.begin(), blob_data_arrary.end(), [](auto& lhs, auto& rhs){
        return lhs.blob->id() < rhs.blob->id();
    });

    // allocate for each storage
    for (auto i : blob_data_arrary) {
        DeviceTensorStorage d_storage = DeviceTensorStorage(cn);
        d_storage.ensure_size(i.blob -> m_size);
        d_storage.copy_from(i.h_storage, i.blob -> m_size);
        i.blob -> m_storage = d_storage.raw_storage();
    }

    // wait copy finish before destructing host values
    cn.sync();
}

void BlobManagerImpl::set_enable(bool flag) {
    m_enable = flag;
}

struct BlobManagerStub : BlobManager {
    void alloc_with_defrag(Blob* blob, size_t size) {
        mgb_assert(0, "prohibited after global variable destruction");
    };
    void register_blob(Blob* blob) {
        mgb_assert(0, "prohibited after global variable destruction");
    };
    void unregister_blob(Blob* blob) {};
    void set_enable(bool flag) {
        mgb_assert(0, "prohibited after global variable destruction");
    };
    void defrag(const CompNode& cn) {
        mgb_assert(0, "prohibited after global variable destruction");
    };
};

BlobManager* BlobManager::inst() {
    static std::aligned_union_t<0, BlobManagerImpl, BlobManagerStub> storage;

    struct Keeper {
        Keeper() {
            new(&storage) BlobManagerImpl();
        }
        ~Keeper() {
            reinterpret_cast<BlobManager*>(&storage)->~BlobManager();
            new(&storage) BlobManagerStub();
        }
    };
    static Keeper _;

    return reinterpret_cast<BlobManager*>(&storage);
}

} // namespace imperative
} // namespace mgb
