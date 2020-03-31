/**
 * \file dnn/include/megcore.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/thin/function.h"
#include "megcore_cdefs.h"
#include <cstddef>
#include <memory>

#include "megdnn/internal/visibility_prologue.h"

namespace megcore {
/*!
 * \brief a callback to dispatch computing task on desired CPU thread
 *
 * This is analogous to cuda streams. The default dispatcher on CPU executes in
 * the caller thread immediately.
 */
class CPUDispatcher {
    public:
        using Task = megdnn::thin_function<void()>;
        using MultiThreadingTask = megdnn::thin_function<void(size_t, size_t)>;
        virtual ~CPUDispatcher() noexcept;

        /*!
         * \brief dispatch a task on the computing thread
         * \param task the task that would be moved away
         */
        virtual void dispatch(Task&& task) = 0;

        /*!
         * \brief dispatch a multithreading task on the computing thread
         * \param task the task would be moved away
         * \param parallelism the parallelism of the task.
         */
        virtual void dispatch(MultiThreadingTask&& task,
                              size_t parallelism) = 0;

        /*!
         * \brief synchronize the calling thread with the computing thread
         */
        virtual void sync() = 0;

        /*!
         * \brief the computing thread number.
         */
        virtual size_t nr_threads() = 0;
};
} // namespace megcore

using MegcoreCPUDispatcher = megcore::CPUDispatcher;

/**
 * \brief Layer 1: device handle
 */
struct megcoreDeviceContext;
typedef struct megcoreDeviceContext *megcoreDeviceHandle_t;

megcoreStatus_t megcoreCreateDeviceHandle(
        megcoreDeviceHandle_t *handle,
        megcorePlatform_t platform,
        int deviceID = -1,
        unsigned int flags = 0);
megcoreStatus_t megcoreDestroyDeviceHandle(
        megcoreDeviceHandle_t handle);

megcoreStatus_t megcoreGetPlatform(megcoreDeviceHandle_t handle,
        megcorePlatform_t *platform);
megcoreStatus_t megcoreGetDeviceID(megcoreDeviceHandle_t handle,
        int *deviceID);
megcoreStatus_t megcoreGetMemAlignment(megcoreDeviceHandle_t handle,
        size_t *memAlignmentInBytes);
megcoreStatus_t megcoreGetDeviceFlags(
        megcoreDeviceHandle_t handle,
        unsigned int *flags);

megcoreStatus_t megcoreActivate(megcoreDeviceHandle_t handle);
megcoreStatus_t megcoreDeactivate(megcoreDeviceHandle_t handle);
megcoreStatus_t megcoreMalloc(megcoreDeviceHandle_t handle,
        void **devPtr, size_t sizeInBytes);
megcoreStatus_t megcoreFree(megcoreDeviceHandle_t handle,
        void *devPtr);

/**
 * \brief Layer 2: computing handle
 */
struct megcoreComputingContext;
typedef struct megcoreComputingContext *megcoreComputingHandle_t;

megcoreStatus_t megcoreCreateComputingHandle(
        megcoreComputingHandle_t *compHandle,
        megcoreDeviceHandle_t devHandle,
        unsigned int flags = 0);

megcoreStatus_t megcoreCreateComputingHandleWithCPUDispatcher(
        megcoreComputingHandle_t *compHandle,
        megcoreDeviceHandle_t devHandle,
        const std::shared_ptr<MegcoreCPUDispatcher>& dispatcher,
        unsigned int flags = 0);

megcoreStatus_t megcoreDestroyComputingHandle(
        megcoreComputingHandle_t handle);

megcoreStatus_t megcoreGetDeviceHandle(
        megcoreComputingHandle_t compHandle,
        megcoreDeviceHandle_t *devHandle);
megcoreStatus_t megcoreGetComputingFlags(
        megcoreComputingHandle_t handle,
        unsigned int *flags);

MegcoreCPUDispatcher* megcoreGetCPUDispatcher(megcoreComputingHandle_t handle);

megcoreStatus_t megcoreMemcpy(
        megcoreComputingHandle_t handle,
        void *dst, const void *src, size_t sizeInBytes,
        megcoreMemcpyKind_t kind);
megcoreStatus_t megcoreMemset(
        megcoreComputingHandle_t handle,
        void *dst, int value, size_t sizeInBytes);
megcoreStatus_t megcoreSynchronize(megcoreComputingHandle_t handle);

/**
 * \brief Miscellaneous
 */
const char *megcoreGetErrorName(megcoreStatus_t status);

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
