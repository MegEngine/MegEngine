/**
 * \file imperative/src/include/megbrain/imperative/blob_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/physical_tensor.h"

namespace mgb {
namespace imperative {

class BlobManager : public NonCopyableObj {
public:
    virtual ~BlobManager() = default;

    static BlobManager* inst();

    virtual void alloc_with_defrag(Blob* blob, size_t size) = 0;

    virtual void register_blob(Blob* blob) = 0;

    virtual void unregister_blob(Blob* blob) = 0;

    virtual void set_enable(bool flag) = 0;

    virtual void defrag(const CompNode& cn) = 0;
};

} // namespace imperative
} // namespace mgb
