/**
 * \file include/megcore_cambricon.h
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 */

#pragma once

#include "megcore.h"

#include <cndev.h>
#include <cnnl.h>
#include <cnrt.h>

#include "megdnn/internal/visibility_prologue.h"

namespace megcore {

struct CambriconMemoryManager {
    /*!
     * \brief alloc function.
     * \param size size to alloc.
     * \return void* logical address on cambricon.
     */
    virtual void* alloc(size_t size) = 0;

    /*!
     * \brief free function.
     * \param ptr logical address to free.
     */
    virtual void free(void* ptr) = 0;
    virtual ~CambriconMemoryManager() = default;
};

typedef CambriconMemoryManager* CambriconMemoryManager_t;

megcoreStatus_t createDeviceHandleWithGlobalInitStatus(
        megcoreDeviceHandle_t* devHandle, int deviceID, unsigned int flags,
        bool global_initialized);

struct CambriconContext {
    CambriconContext() = default;
    CambriconContext(cnrtQueue_t q, cnnlHandle_t h, CambriconMemoryManager_t m)
            : queue{q}, cnnl_handle{h}, mem_mgr{m} {}
    cnrtQueue_t queue = nullptr;
    cnnlHandle_t cnnl_handle = nullptr;
    CambriconMemoryManager_t mem_mgr = nullptr;
};

megcoreStatus_t createComputingHandleWithCambriconContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const CambriconContext& ctx);

megcoreStatus_t getCambriconContext(
        megcoreComputingHandle_t handle, CambriconContext* ctx);

}  // namespace megcore

static inline megcoreStatus_t megcoreCreateComputingHandleWithCNRTQueue(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, cnrtQueue_t queue, cnnlHandle_t cnnl_handle,
        megcore::CambriconMemoryManager_t mem_mgr) {
    megcore::CambriconContext ctx{queue, cnnl_handle, mem_mgr};
    return megcore::createComputingHandleWithCambriconContext(
            compHandle, devHandle, flags, ctx);
}

static inline megcoreStatus_t megcoreGetCNRTQueue(
        megcoreComputingHandle_t handle, cnrtQueue_t* queue) {
    megcore::CambriconContext ctx;
    auto ret = megcore::getCambriconContext(handle, &ctx);
    *queue = ctx.queue;
    return ret;
}

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
