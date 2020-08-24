/**
 * \file dnn/src/cambricon/megcore/cambricon_computing_context.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megcore.h"

#include "src/cambricon/utils.h"
#include "src/common/utils.h"

#include "src/cambricon/megcore/cambricon_computing_context.hpp"

using namespace megcore;
using namespace megcore::cambricon;

CambriconComputingContext::CambriconComputingContext(
        megcoreDeviceHandle_t dev_handle, unsigned int flags,
        const CambriconContext& ctx)
        : ComputingContext(dev_handle, flags),
          own_queue{ctx.queue == nullptr},
          context_{ctx} {
    megcorePlatform_t platform;
    megcoreGetPlatform(dev_handle, &platform);
    megdnn_assert(platform == megcorePlatformCambricon);
    if (own_queue) {
        cnrt_check(cnrtCreateQueue(&context_.queue));
    }
}

CambriconComputingContext::~CambriconComputingContext() {
    if (own_queue) {
        cnrt_check(cnrtDestroyQueue(context_.queue));
    }
}

void CambriconComputingContext::memcpy(void* dst, const void* src,
                                       size_t size_in_bytes,
                                       megcoreMemcpyKind_t kind) {
    cnrtMemTransDir_t dir;
    switch (kind) {
        case megcoreMemcpyDeviceToHost:
            dir = CNRT_MEM_TRANS_DIR_DEV2HOST;
            break;
        case megcoreMemcpyHostToDevice:
            dir = CNRT_MEM_TRANS_DIR_HOST2DEV;
            break;
        case megcoreMemcpyDeviceToDevice:
            dir = CNRT_MEM_TRANS_DIR_DEV2DEV;
            break;
        default:
            megdnn_throw(megdnn_mangle("bad cnrt mem trans dir"));
    }
    if (kind == megcoreMemcpyDeviceToDevice) {
        cnrt_check(cnrtSyncQueue(context_.queue));
        cnrt_check(cnrtMemcpy(dst, const_cast<void*>(src), size_in_bytes, dir));
        return;
    }
    cnrt_check(cnrtMemcpyAsync(dst, const_cast<void*>(src), size_in_bytes,
                               context_.queue, dir));
}

void CambriconComputingContext::memset(void* dst, int value,
                                       size_t size_in_bytes) {
    cnrt_check(cnrtSyncQueue(context_.queue));
    cnrt_check(cnrtMemset(dst, value, size_in_bytes));
}

void CambriconComputingContext::synchronize() {
    cnrt_check(cnrtSyncQueue(context_.queue));
}

// vim: syntax=cpp.doxygen

