/**
 * \file include/megcore_cambricon.h
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

#pragma once

#include "megcore.h"

#include <cndev.h>
#include <cnml.h>
#include <cnrt.h>

#include "megdnn/internal/visibility_prologue.h"

namespace megcore {
megcoreStatus_t createDeviceHandleWithGlobalInitStatus(
        megcoreDeviceHandle_t* devHandle, int deviceID, unsigned int flags,
        bool global_initialized);

struct CambriconContext {
    cnrtQueue_t queue = nullptr;

    CambriconContext() = default;

    CambriconContext(cnrtQueue_t q) : queue{q} {}
};

megcoreStatus_t createComputingHandleWithCambriconContext(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, const CambriconContext& ctx);

megcoreStatus_t getCambriconContext(megcoreComputingHandle_t handle,
                                    CambriconContext* ctx);

}  // namespace megcore

static inline megcoreStatus_t megcoreCreateComputingHandleWithCNRTQueue(
        megcoreComputingHandle_t* compHandle, megcoreDeviceHandle_t devHandle,
        unsigned int flags, cnrtQueue_t queue) {
    megcore::CambriconContext ctx{queue};
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

