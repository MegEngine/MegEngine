/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
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
