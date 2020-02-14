/**
 * \file dnn/test/common/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "src/common/utils.h"

#include <memory>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>

#if MEGDNN_ENABLE_MULTI_THREADS
#include <atomic>
#endif

#define megcore_check(x)                                           \
    do {                                                           \
        auto status = (x);                                         \
        if (status != megcoreSuccess) {                            \
            std::cerr << "megcore_check error: "                   \
                      << megcoreGetErrorName(status) << std::endl; \
            megdnn_trap();                                         \
        }                                                          \
    } while (0)

namespace megdnn {
namespace test {

struct TaskExecutorConfig {
    //! Number of threads.
    size_t nr_thread;
    //! The core id to bind. The size of affinity_core_set should be equal to
    //! nr_thread.
    std::vector<size_t> affinity_core_set;
};

class CpuDispatchChecker final : MegcoreCPUDispatcher {
    class TaskExecutor {
        using Task = megcore::CPUDispatcher::Task;
        using MultiThreadingTask = megcore::CPUDispatcher::MultiThreadingTask;
#if MEGDNN_ENABLE_MULTI_THREADS
#if defined(WIN32)
        using thread_affinity_type = DWORD;
#else  // not WIN32
#if defined(__APPLE__)
        using thread_affinity_type = int;
#else
        using thread_affinity_type = cpu_set_t;
#endif
#endif
#endif

    public:
        TaskExecutor(TaskExecutorConfig* config = nullptr);
        ~TaskExecutor();
        /*!
         * Sync all workers.
         */
        void sync();
        /*!
         * Number of threads in this thread pool, including the main thread.
         */
        size_t nr_threads() const { return m_nr_threads; }
        void add_task(const MultiThreadingTask& task, size_t parallelism);
        void add_task(const Task& task);

    private:
#if MEGDNN_ENABLE_MULTI_THREADS
        size_t m_all_task_iter = 0;
        std::atomic_int m_current_task_iter{0};

        //! Indicate whether the thread should work, used for main thread sync
        std::vector<std::atomic_bool*> m_workers_flag;

        //! Whether the main thread affinity has been set.
        bool m_main_thread_affinity = false;

        //! Stop the worker threads.
        bool m_stop{false};

        MultiThreadingTask m_task;

        //! The cpuids to be bound.
        //! If the m_cpu_ids is empty, then none of the threads will be bound to
        //! cpus, else the size of m_cpu_ids should equal to m_nr_threads.
        std::vector<size_t> m_cpu_ids;

        //! The previous affinity mask of the main thread.
        thread_affinity_type m_main_thread_prev_affinity_mask;

        std::vector<std::thread> m_workers;
#endif
        //! Total number of threads, including main thread.
        size_t m_nr_threads = 0;
    };

    //! track number of CpuDispatchChecker instances to avoid leaking
    class InstCounter {
        bool m_used = false;
        int m_cnt = 0, m_max_cnt = 0;

    public:
        ~InstCounter() {
            auto check = [this]() {
                ASSERT_NE(0, m_max_cnt) << "no kernel dispatched on CPU";
                ASSERT_EQ(0, m_cnt) << "leaked CpuDispatchChecker object";
            };
            if (m_used) {
                check();
            }
        }
        int& cnt() {
            m_used = true;
            m_max_cnt = std::max(m_cnt, m_max_cnt);
            return m_cnt;
        }
    };
    static InstCounter sm_inst_counter;
    bool m_recursive_dispatch = false;
#if MEGDNN_ENABLE_MULTI_THREADS
    std::atomic_size_t m_nr_call{0};
#else
    size_t m_nr_call = 0;
#endif

    std::unique_ptr<TaskExecutor> m_task_executor;

    CpuDispatchChecker(TaskExecutorConfig* config) {
        ++sm_inst_counter.cnt();
        megdnn_assert(sm_inst_counter.cnt() < 10);
        m_task_executor = std::make_unique<TaskExecutor>(config);
    }

    void dispatch(Task&& task) override {
        megdnn_assert(!m_recursive_dispatch);
        m_recursive_dispatch = true;
        ++m_nr_call;
        m_task_executor->add_task(std::move(task));
        m_recursive_dispatch = false;
    }

    void dispatch(MultiThreadingTask&& task, size_t parallelism) override {
        megdnn_assert(!m_recursive_dispatch);
        m_recursive_dispatch = true;
        ++m_nr_call;
        m_task_executor->add_task(std::move(task), parallelism);
        m_recursive_dispatch = false;
    }

    size_t nr_threads() override { return m_task_executor->nr_threads(); }

    CpuDispatchChecker() {
        ++sm_inst_counter.cnt();
        megdnn_assert(sm_inst_counter.cnt() < 10);
    }

    void sync() override {}

public:
    ~CpuDispatchChecker() {
        if (!std::uncaught_exception()) {
            megdnn_assert(!m_recursive_dispatch);
#if !MEGDNN_NO_THREAD
            megdnn_assert(m_nr_call && "cpu dispatch must be called");
#endif
        } else {
            if (m_recursive_dispatch) {
                fprintf(stderr,
                        "CpuDispatchChecker: "
                        "detected recursive dispatch\n");
            }
            if (!m_nr_call) {
                fprintf(stderr, "CpuDispatchChecker: dispatch not called\n");
            }
        }
        --sm_inst_counter.cnt();
    }

    static std::unique_ptr<MegcoreCPUDispatcher> make(
            TaskExecutorConfig* config) {
        return std::unique_ptr<MegcoreCPUDispatcher>(
                new CpuDispatchChecker(config));
    }
};

std::unique_ptr<Handle> create_cpu_handle(int debug_level,
                                          bool check_dispatch = true,
                                          TaskExecutorConfig* config = nullptr);

std::unique_ptr<Handle> create_cpu_handle_with_dispatcher(
        int debug_level,
        const std::shared_ptr<MegcoreCPUDispatcher>& dispatcher);

static inline dt_float32 diff(dt_float32 x, dt_float32 y) {
    auto numerator = x - y;
    auto denominator = std::max(std::max(std::abs(x), std::abs(y)), 1.f);
    return numerator / denominator;
}

static inline int diff(int x, int y) {
    return x - y;
}

static inline int diff(dt_quint8 x, dt_quint8 y) {
    return x.as_uint8() - y.as_uint8();
}

static inline int diff(dt_qint32 x, dt_qint32 y) {
    return x.as_int32() - y.as_int32();
}

static inline int diff(dt_qint16 x, dt_qint16 y) {
    return x.as_int16() - y.as_int16();
}

static inline int diff(dt_qint8 x, dt_qint8 y) {
    return x.as_int8() - y.as_int8();
}

inline TensorShape cvt_src_or_dst_nchw2nhwc(const TensorShape& shape) {
    megdnn_assert(shape.ndim == 4);
    auto N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    return TensorShape{N, H, W, C};
}

inline TensorShape cvt_src_or_dst_ncdhw2ndhwc(const TensorShape& shape) {
    megdnn_assert(shape.ndim == 5);
    auto N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];
    return TensorShape{N, D, H, W, C};
}

inline TensorShape cvt_filter_nchw2nhwc(const TensorShape& shape) {
    if (shape.ndim == 4) {
        auto OC = shape[0], IC = shape[1], FH = shape[2], FW = shape[3];
        return TensorShape{OC, FH, FW, IC};
    } else {
        megdnn_assert(shape.ndim == 5);
        auto G = shape[0], OC = shape[1], IC = shape[2], FH = shape[3],
             FW = shape[4];
        return TensorShape{G, OC, FH, FW, IC};
    }
}

inline TensorShape cvt_filter_ncdhw2ndhwc(const TensorShape& shape) {
    if (shape.ndim == 5) {
        auto OC = shape[0], IC = shape[1], FD = shape[2], FH = shape[3],
             FW = shape[4];
        return TensorShape{OC, FD, FH, FW, IC};
    } else {
        megdnn_assert(shape.ndim == 6);
        auto G = shape[0], OC = shape[1], IC = shape[2], FD = shape[3],
             FH = shape[4], FW = shape[5];
        return TensorShape{G, OC, FD, FH, FW, IC};
    }
}

void megdnn_sync(Handle* handle);
void* megdnn_malloc(Handle* handle, size_t size_in_bytes);
void megdnn_free(Handle* handle, void* ptr);
void megdnn_memcpy_D2H(Handle* handle, void* dst, const void* src,
                       size_t size_in_bytes);
void megdnn_memcpy_H2D(Handle* handle, void* dst, const void* src,
                       size_t size_in_bytes);
void megdnn_memcpy_D2D(Handle* handle, void* dst, const void* src,
                       size_t size_in_bytes);

//! default implementation for DynOutMallocPolicy
class DynOutMallocPolicyImpl final : public DynOutMallocPolicy {
    Handle* m_handle;

public:
    DynOutMallocPolicyImpl(Handle* handle) : m_handle{handle} {}

    TensorND alloc_output(size_t id, DType dtype, const TensorShape& shape,
                          void* user_data) override;
    void* alloc_workspace(size_t sz, void* user_data) override;
    void free_workspace(void* ptr, void* user_data) override;

    /*!
     * \brief make a shared_ptr which would release output memory when
     *      deleted
     * \param out output tensor allocated by alloc_output()
     */
    std::shared_ptr<void> make_output_refholder(const TensorND& out);
};

//! replace ErrorHandler::on_megdnn_error
class MegDNNError : public std::exception {
    std::string m_msg;

public:
    MegDNNError(const std::string& msg) : m_msg{msg} {}

    const char* what() const noexcept { return m_msg.c_str(); }
};
class TensorReshapeError : public MegDNNError {
public:
    using MegDNNError::MegDNNError;
};

size_t get_cpu_count();

}  // namespace test

static inline bool operator==(const TensorLayout& a, const TensorLayout& b) {
    return a.eq_layout(b);
}

static inline std::ostream& operator<<(std::ostream& ostr,
                                       const TensorLayout& layout) {
    return ostr << layout.to_string();
}

//! change the image2d_pitch_alignment of naive handle in this scope
class NaivePitchAlignmentScope {
    size_t m_orig_val, m_new_val;

public:
    NaivePitchAlignmentScope(size_t alignment);
    ~NaivePitchAlignmentScope();
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
