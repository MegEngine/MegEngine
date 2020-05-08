/**
 * \file src/core/include/megbrain/comp_node/alloc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node_env.h"
#include "megbrain/exception.h"
#include "megbrain/common.h"

namespace mgb {
namespace mem_alloc {

/*!
 * \brief interface for raw allocator
 *
 * In case of allocation error, MemAllocError should be thrown
 */
class RawAllocator {
    public:
        /*!
         * \brief allocate memory of requested size (in bytes); if memory is run
         *      out, return nullptr; in other cases of error, throw
         *      MemAllocError
         */
        virtual void* alloc(size_t size) = 0;

        /*!
         * \brief release allocated memory
         */
        virtual void free(void *addr) = 0;

        /*!
         * \brief get free and total device memory
         * \param[out] free returned free memory in bytes
         * \param[out] tot returned total memory in bytes
         */
        virtual void get_mem_info(size_t& free, size_t& tot) = 0;

        virtual ~RawAllocator() = default;
};

/*!
 * \brief like RawAllocator, but alloc() would not return nullptr; an exception
 *      is thrown if fails
 */
class NonFallibleRawAllocator: public RawAllocator {
    public:

        /*!
         * \brief allocate as shared_ptr that binds the deallocator
         */
        virtual std::shared_ptr<void> alloc_shared(size_t size) {
            auto del = [this](void* p) { free(p); };
            return {alloc(size), del};
        }
};

/*!
 * \brief statistics about free memory
 */
struct FreeMemStat {
    size_t tot, min, max, nr_blk;
};

/*!
 * \brief interface for device runtime policy
 */
class DeviceRuntimePolicy {
public:
    /*!
     * \brief return the device type of this runtime policy
     * \return CompNode::DeviceType
     */
    virtual CompNode::DeviceType device_type() = 0;

    /*!
     * \brief set device to be used for GPU executions
     * \param[in] device Device on which the active host thread should execute
     *      the device code
     */
    virtual void set_device(int device) = 0;

    /*!
     * \brief block the calling thread until the device corresponding to the
     *      given device has completed all preceding requested tasks.
     *
     * Note:
     * 1. This interface must be implemented with the device driver/runtime API,
     *    e.g. cudaDeviceSynchronize. This interface is different from
     *    CompNode::sync() which is based on Event.
     * 2. For CUDA implementation, user should set device to the given device
     *    and do the synchronization. This can be done by using the CUDA Runtime
     *    API cudaSetDevice/cudaDeviceSynchronize.
     * \param[in] device Device on which the active host thread should execute
     *      the device CompNode
     */
    virtual void device_synchronize(int device) = 0;

    virtual ~DeviceRuntimePolicy() = default;
};

class MemAllocBase {
    public:
        /*!
         * \brief print current memory state to log
         */
        virtual void print_memory_state() = 0;

        /*!
         * \brief get total size of allocated memory
         */
        virtual size_t get_used_memory() = 0;

        /*!
         * \brief get free memory stats on current allocator
         *
         * \see get_free_memory_dev
         */
        virtual FreeMemStat get_free_memory() = 0;

        /*!
         * \brief get free memory on the whole device
         *
         * All stream allocators and device allocator on the same device are
         * considered.
         */
        virtual FreeMemStat get_free_memory_dev() = 0;

        virtual ~MemAllocBase() = default;
};

class StreamMemAlloc: virtual public NonFallibleRawAllocator,
                      virtual public MemAllocBase {
    public:

        /*!
         * \brief allocate memory
         *
         *  Note that the caller is responsible to call cudaSetDevice before
         *  calling.
         */
        virtual void* alloc(size_t size) = 0;

        virtual void free(void *addr) = 0;
};

/*!
 * \brief dynamic memory allocator on a device
 *
 * It has a two-level structure, where the root allocator requests memory from a
 * user-supplied RawAllocator, and the children allocator lives on a stream,
 * which maintains a local pool of memory.
 *
 * All methods are thread safe.
 */
class DevMemAlloc: virtual public MemAllocBase {
    public:
        using StreamKey = void*;

        /*!
         * \brief specifies how to pre-allocate from raw dev allocator
         */
        struct PreAllocConfig {
            static constexpr size_t MB = 1024 * 1024;

            double growth_factor = 2;       //! req size / cur allocated
            size_t
                min_req = 32 * MB,          //! min request to raw allocator
                max_overhead = 256 * MB,    //! max overhead (above asked size)
                alignment = 1024;           //! alignment
        };

        /*!
         * \brief create a new allocator for a device
         * \param[in] device device id
         * \param[in] reserve_size memory to be pre-allocated on this device
         * \param[in] raw_allocator the raw allocator to be used
         * \param[in] runtime_policy the runtime policy to be used
         */
        static std::unique_ptr<DevMemAlloc> make(
                int device, size_t reserve_size,
                const std::shared_ptr<mem_alloc::RawAllocator>& raw_allocator,
                const std::shared_ptr<mem_alloc::DeviceRuntimePolicy>&
                        runtime_policy);

#if MGB_CUDA
        /*!
         * \brief create a new allocator for a device that merly forward
         *      cudaMalloc and cudaFree, so no custom algorithm is involved
         */
        static std::unique_ptr<DevMemAlloc> make_cuda_alloc();
#endif


        virtual ~DevMemAlloc() = default;

        /*!
         * \brief gather all free blocks from child streams, and release full
         *      chunks back to parent allocator
         * \return number of bytes released
         */
        virtual size_t gather_stream_free_blk_and_release_full() = 0;

        /*!
         * \brief create a child allocator on a stream; its lifespan is the same
         *      as this DevMemAlloc
         */
        virtual StreamMemAlloc* add_stream(StreamKey stream) = 0;

        /*!
         * \brief get the underlying raw allocator
         */
        virtual const std::shared_ptr<RawAllocator>& raw_allocator() const = 0;

        /*!
         * \brief get the underlying device runtime policy
         */
        virtual const std::shared_ptr<DeviceRuntimePolicy>& device_runtime_policy() const = 0;

        /*!
         * \brief set alignment of allocated addresses
         * \param alignment desired alignment, which must be a power of 2
         */
        DevMemAlloc& alignment(size_t alignment) {
            mgb_assert(alignment && !(alignment & (alignment - 1)));
            m_alignment = alignment;
            return *this;
        }

        /*!
         * \brief set prealloc config
         */
        DevMemAlloc& prealloc_config(const PreAllocConfig &conf) {
            mgb_assert(conf.alignment &&
                    !(conf.alignment & (conf.alignment - 1)));
            m_prealloc_config = conf;
            return *this;
        }

        /*!
         * \brief get current alignment
         */
        size_t alignment() const {
            return m_alignment;
        }

        const PreAllocConfig& prealloc_config() {
            return m_prealloc_config;
        }

    private:
        size_t m_alignment = 1;
        PreAllocConfig m_prealloc_config;
};

/* ===================== FwdDevMemAlloc  ===================== */
/*!
 * \brief Allocator for a device that merely forward alloc/free provided by the
 * device runtime api. No custom algorithm is involved. This class will be used
 * by make_cuda_alloc.
 */
class FwdDevMemAlloc final : public DevMemAlloc {
    class StreamMemAllocImpl final : public StreamMemAlloc {
        FwdDevMemAlloc* const m_par_alloc;

        void* alloc(size_t size) override {
            auto ptr = m_par_alloc->m_raw_alloc->alloc(size);
            mgb_throw_if(!ptr, MemAllocError, "failed to alloc %zu bytes",
                         size);
            return ptr;
        }

        void free(void* addr) override { m_par_alloc->m_raw_alloc->free(addr); }

        void get_mem_info(size_t& free, size_t& tot) override {
            m_par_alloc->m_raw_alloc->get_mem_info(free, tot);
        }

        void print_memory_state() override {}

        size_t get_used_memory() override { mgb_assert(0); }

        FreeMemStat get_free_memory() override { mgb_assert(0); }

        FreeMemStat get_free_memory_dev() override {
            size_t tot, free;
            m_par_alloc->m_raw_alloc->get_mem_info(free, tot);
            return {free, free, free, 1};
        }

    public:
        StreamMemAllocImpl(FwdDevMemAlloc* par_alloc)
                : m_par_alloc(par_alloc) {}
    };

    std::mutex m_mtx;
    std::shared_ptr<RawAllocator> m_raw_alloc;
    std::shared_ptr<DeviceRuntimePolicy> m_runtime_policy;
    ThinHashMap<StreamKey, std::unique_ptr<StreamMemAllocImpl>> m_stream_alloc;

    void print_memory_state() override {}

    size_t get_used_memory() override { mgb_assert(0); }

    FreeMemStat get_free_memory() override { mgb_assert(0); }

    FreeMemStat get_free_memory_dev() override {
        size_t tot, free;
        m_raw_alloc->get_mem_info(free, tot);
        return {free, free, free, 1};
    }

    StreamMemAlloc* add_stream(StreamKey stream) override {
        MGB_LOCK_GUARD(m_mtx);
        auto&& v = m_stream_alloc[stream];
        if (!v)
            v = std::make_unique<StreamMemAllocImpl>(this);
        return v.get();
    }

    const std::shared_ptr<RawAllocator>& raw_allocator() const override {
        return m_raw_alloc;
    }

    const std::shared_ptr<DeviceRuntimePolicy>& device_runtime_policy()
            const override {
        return m_runtime_policy;
    }

    size_t gather_stream_free_blk_and_release_full() override { return 0; }

public:
    FwdDevMemAlloc(const std::shared_ptr<RawAllocator>& ra) : m_raw_alloc(ra) {}
};

} // mem_alloc
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
