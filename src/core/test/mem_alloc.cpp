/**
 * \file src/core/test/mem_alloc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#include "megbrain/comp_node/alloc.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"

#include <thread>
#include <map>
#include <random>
#include <atomic>

using namespace mgb;
using namespace mem_alloc;

namespace {
class DummyRuntimePolicy final : public DeviceRuntimePolicy {
    int m_device;

public:
    explicit DummyRuntimePolicy(int device) : m_device{device} {}
    void set_device(int device) override { m_device = device; }
    void device_synchronize(int /* device */) override {}
    CompNode::DeviceType device_type() override {
        return CompNode::DeviceType::CPU;
    }
};

class DummyAllocator final: public RawAllocator {
    const size_t m_tot_size;
    bool m_ever_failed = false;
    size_t m_next_addr = 1, m_cur_usage = 0, m_peak_usage = 0,
           m_nr_alloc = 0, m_nr_free = 0;
    std::map<void*, size_t> m_addr2size;
    std::mutex m_mtx;

    public:
        explicit DummyAllocator(size_t tot_size):
            m_tot_size(tot_size)
        {}

        ~DummyAllocator()
        {
            auto run = [this]() {
                ASSERT_EQ(0u, m_addr2size.size());
            };
            run();
        }

        void* alloc(size_t size) override {
            MGB_LOCK_GUARD(m_mtx);
            if (mgb_unlikely(m_cur_usage + size > m_tot_size)) {
                m_ever_failed = true;
                return nullptr;
            }
            ++m_nr_alloc;
            auto addr = reinterpret_cast<void*>(m_next_addr);
            m_next_addr += size;
            m_cur_usage += size;
            m_peak_usage = std::max(m_peak_usage, m_cur_usage);
            m_addr2size[addr] = size;
            return addr;
        }

        void free(void *ptr) override {
            MGB_LOCK_GUARD(m_mtx);
            auto iter = m_addr2size.find(ptr);
            mgb_assert(iter != m_addr2size.end());
            ++ m_nr_free;
            m_cur_usage -= iter->second;
            m_addr2size.erase(iter);
        }

        void get_mem_info(size_t& free, size_t& tot) override {
            tot = m_tot_size;
            free = free_size();
        }

        size_t free_size() const {
            return m_tot_size - m_cur_usage;
        }

        bool ever_failed() const {
            return m_ever_failed;
        }

        size_t peak_usage() const {
            return m_peak_usage;
        }

        size_t nr_alloc() const {
            return m_nr_alloc;
        }

        size_t nr_free() const {
            return m_nr_free;
        }

        void* get_chunk_end(void *addr) {
            MGB_LOCK_GUARD(m_mtx);
            auto iter = m_addr2size.upper_bound(addr);
            mgb_assert(iter != m_addr2size.begin() &&
                    (iter == m_addr2size.end() || iter->first > addr));
            -- iter;
            void* end = (char*)iter->first + iter->second;
            mgb_assert(iter->first <= addr && end > addr);
            return end;
        }

};

class AllocChecker {
    std::shared_ptr<DummyAllocator> m_root_allocator;
    size_t m_peak_usage = 0, m_cur_usage = 0;
    std::map<size_t, size_t> m_addr2size;
    std::mutex m_mtx;

    public:

        AllocChecker(std::shared_ptr<DummyAllocator> root_alloc):
            m_root_allocator(std::move(root_alloc))
        {}

        void add(void *addr_, size_t size) {
            ASSERT_NE(nullptr, addr_);
            mgb_assert((char*)addr_ + size <=
                    m_root_allocator->get_chunk_end(addr_));
            auto addr = reinterpret_cast<size_t>(addr_);
            MGB_LOCK_GUARD(m_mtx);
            auto rst = m_addr2size.insert({addr, size});
            mgb_assert(rst.second, "duplicated address: %p", addr_);
            auto iter = rst.first;
            if (mgb_likely(iter != m_addr2size.begin())) {
                auto iprev = iter;
                -- iprev;
                mgb_assert(iprev->first + iprev->second <= addr);
            }
            auto inext = iter;
            ++ inext;
            if (mgb_likely(inext != m_addr2size.end())) {
                mgb_assert(addr + size <= inext->first);
            }

            m_cur_usage += size;
            m_peak_usage = std::max(m_peak_usage, m_cur_usage);
        }

        void remove(void *addr) {
            MGB_LOCK_GUARD(m_mtx);
            auto iter = m_addr2size.find(reinterpret_cast<size_t>(addr));
            mgb_assert(iter != m_addr2size.end());
            m_cur_usage -= iter->second;
            m_addr2size.erase(iter);
        }

        size_t peak_usage() const {
            return m_peak_usage;
        }
};

} // anonymous namespace

TEST(TestMemAlloc, Reserve) {
    constexpr size_t TOT = 2048;

    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, TOT, raw_alloc, runtime_policy);

    StreamKey stream_key = nullptr;
    auto strm_alloc =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));
    EXPECT_EQ(0u, strm_alloc->get_free_memory().tot);
    EXPECT_EQ(2048u, dev_alloc->get_free_memory().tot);
}

TEST(TestMemAlloc, ReserveOutOfMemory) {
    constexpr size_t TOT = 2048;

    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    EXPECT_THROW(DevMemAlloc::make(0, TOT + 1, raw_alloc, runtime_policy),
                 MemAllocError);
}

TEST(TestMemAlloc, Alloc) {
    constexpr size_t TOT = 2048, REQ = 1000;
    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, TOT, raw_alloc, runtime_policy);

    StreamKey stream_key = nullptr;
    auto strm_alloc =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));

    auto ptr = strm_alloc->alloc_shared(REQ);
    EXPECT_EQ(REQ, strm_alloc->get_used_memory());
    EXPECT_EQ(TOT - REQ, strm_alloc->get_free_memory().tot);
    EXPECT_EQ(TOT, dev_alloc->get_used_memory());
    EXPECT_EQ(0u, dev_alloc->get_free_memory().tot);
    auto addr = ptr.get();
    ptr.reset();
    EXPECT_EQ(0u, strm_alloc->get_used_memory());
    EXPECT_EQ(TOT, strm_alloc->get_free_memory().tot);
    EXPECT_EQ(TOT, dev_alloc->get_used_memory());
    EXPECT_EQ(0u, dev_alloc->get_free_memory().tot);
    EXPECT_EQ(addr, strm_alloc->alloc_shared(REQ).get());
}

TEST(TestMemAlloc, MergeFreeBlock) {
    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(7000);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, 7000, raw_alloc, runtime_policy);

    StreamKey stream_key = nullptr;
    auto strm_alloc =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));

    auto ptr = strm_alloc->alloc_shared(2000);
    auto addr = ptr.get();
    ptr.reset();
    ptr = strm_alloc->alloc_shared(3000);
    EXPECT_EQ(addr, ptr.get());
    strm_alloc->alloc_shared(4000);
}

TEST(TestMemAlloc, AllocTwoStream) {
    constexpr size_t TOT = 2048, REQ0 = 1000, REQ1 = 2000;
    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, TOT, raw_alloc, runtime_policy);

    StreamKey stream_key0, stream_key1;
    auto strm_alloc0 =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key0)),
         strm_alloc1 =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key1));
    ASSERT_NE(strm_alloc0, strm_alloc1);

    auto ptr0 = strm_alloc0->alloc_shared(REQ0);
    EXPECT_EQ(REQ0, strm_alloc0->get_used_memory());
    EXPECT_EQ(0u, strm_alloc0->get_free_memory().tot);
    EXPECT_EQ(REQ0, dev_alloc->get_used_memory());
    EXPECT_EQ(TOT - REQ0, dev_alloc->get_free_memory().tot);
    ptr0.reset();
    EXPECT_EQ(0u, strm_alloc0->get_used_memory());
    EXPECT_EQ(REQ0, strm_alloc0->get_free_memory().tot);
    EXPECT_EQ(REQ0, dev_alloc->get_used_memory());
    EXPECT_EQ(TOT - REQ0, dev_alloc->get_free_memory().tot);
    auto ptr1 = strm_alloc1->alloc_shared(REQ1);
    EXPECT_EQ(0u, strm_alloc0->get_free_memory().tot);
    EXPECT_EQ(REQ1, strm_alloc1->get_used_memory());
    EXPECT_EQ(0u, strm_alloc1->get_free_memory().tot);
    EXPECT_EQ(REQ1, dev_alloc->get_used_memory());
    EXPECT_EQ(0u, dev_alloc->get_free_memory().tot);
    ptr1.reset();
    EXPECT_EQ(0u, strm_alloc1->get_used_memory());
    EXPECT_EQ(REQ1, strm_alloc1->get_free_memory().tot);
    EXPECT_EQ(REQ1, dev_alloc->get_used_memory());
    EXPECT_EQ(0u, dev_alloc->get_free_memory().tot);
}

TEST(TestMemAlloc, AllocMoreThanReserve) {
    constexpr size_t RES = 1000, TOT = 2048, REQ = 2048;

    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, RES, raw_alloc, runtime_policy);

    StreamKey stream_key = nullptr;
    auto strm_alloc =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));

    auto ptr = strm_alloc->alloc_shared(REQ);
    EXPECT_EQ(REQ, strm_alloc->get_used_memory());
    EXPECT_EQ(0u, strm_alloc->get_free_memory().tot);
    EXPECT_EQ(REQ, dev_alloc->get_used_memory());
    EXPECT_EQ(TOT - REQ, dev_alloc->get_free_memory().tot);
    auto addr = ptr.get();
    ptr.reset();
    EXPECT_EQ(0u, strm_alloc->get_used_memory());
    EXPECT_EQ(REQ, strm_alloc->get_free_memory().tot);
    EXPECT_EQ(REQ, dev_alloc->get_used_memory());
    EXPECT_EQ(TOT - REQ, dev_alloc->get_free_memory().tot);
    EXPECT_EQ(addr, strm_alloc->alloc_shared(REQ).get());
}

TEST(TestMemAlloc, AllocZeroSize) {
    constexpr size_t TOT = 1000;

    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(TOT);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, 1, raw_alloc, runtime_policy);

    StreamKey stream_key = nullptr;
    auto strm_alloc =
            dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));

    EXPECT_ANY_THROW(strm_alloc->alloc(0));
}

TEST(TestMemAlloc, NotCrossBoundary) {
    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(4);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, 0, raw_alloc, runtime_policy);
    auto conf = dev_alloc->prealloc_config();
    conf.max_overhead = 0;
    conf.alignment = 1;
    dev_alloc->prealloc_config(conf);

    StreamKey stream_key = nullptr;
    auto salloc = dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));
    auto p0 = salloc->alloc(1), p1 = salloc->alloc(1);
    salloc->free(p0);
    salloc->free(p1);
    auto p2 = salloc->alloc(2);

    salloc->print_memory_state();
    ASSERT_LE((void*)((char*)p2 + 2), raw_alloc->get_chunk_end(p2)) <<
        p0 << " " << p1 << " " << p2;
}

TEST(TestMemAlloc, GrowByGather) {
    using StreamKey = DevMemAlloc::StreamKey;
    auto raw_alloc = std::make_shared<DummyAllocator>(12);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);
    auto dev_alloc = DevMemAlloc::make(0, 0, raw_alloc, runtime_policy);
    auto conf = dev_alloc->prealloc_config();
    conf.max_overhead = 2;
    conf.alignment = 1;
    dev_alloc->prealloc_config(conf);

    StreamKey stream_key;
    auto salloc = dev_alloc->add_stream(static_cast<StreamKey>(&stream_key));
    salloc->alloc_shared(4);
    salloc->alloc_shared(8);
    salloc->alloc_shared(10);
}

TEST(TestMemAlloc, RandomOprs) {
    const size_t DEALLOC_PROB = std::mt19937::max() * 0.4;
    constexpr size_t NR_THREAD = 4, NR_RUN = 2000, MIN_REQ = 1, MAX_REQ = 513,

                     MAX_MEMORY = NR_THREAD *
                                  (MIN_REQ + (MAX_REQ - MIN_REQ) * 0.5) *
                                  NR_RUN * 0.3,

                     RESERVE_MEMORY = MAX_MEMORY / NR_THREAD * 0.7;

    auto dummy_alloc = std::make_shared<DummyAllocator>(MAX_MEMORY);
    auto runtime_policy = std::make_shared<DummyRuntimePolicy>(0);

    AllocChecker checker(dummy_alloc);
    auto dev_alloc =
            DevMemAlloc::make(0, RESERVE_MEMORY, dummy_alloc, runtime_policy);
    {
        DevMemAlloc::PreAllocConfig prconf;
        prconf.alignment = 512;
        prconf.max_overhead = 0;
        dev_alloc->prealloc_config(prconf);
    }

    std::mt19937 rng_seed(next_rand_seed());
    std::mutex mutex;

    std::atomic_bool start_signal{false}, worker_finished[NR_THREAD];
    std::atomic_int nr_ready_start{0};
    for (auto&& i : worker_finished) {
        i.store(false);
    }

    std::string failed_msg;

    size_t dummy_alloc_peak_usage = 0, checker_peak_usage = 0;
    auto worker_impl = [&](size_t thread_num) {
        std::mt19937 rng;
        {
            MGB_LOCK_GUARD(mutex);
            rng.seed(rng_seed());
        }
        std::vector<std::shared_ptr<void>> allocated_ptrs;
        allocated_ptrs.reserve(NR_RUN);

        ++nr_ready_start;
        while (!start_signal.load())
            ;
        auto stream_alloc = dev_alloc->add_stream(
                reinterpret_cast<DevMemAlloc::StreamKey>(thread_num * 8));

        auto stream_free = [&checker, stream_alloc](void* ptr) {
            checker.remove(ptr);
            stream_alloc->free(ptr);
        };

        for (size_t i = 0; i < NR_RUN; ++i) {
            auto rand_f = rng() / (rng.max() + 1.0);
            if (!allocated_ptrs.empty() && rng() < DEALLOC_PROB) {
                size_t idx = allocated_ptrs.size() * rand_f;
                std::swap(allocated_ptrs.at(idx), allocated_ptrs.back());
                allocated_ptrs.pop_back();
            } else {
                size_t size = (MAX_REQ - MIN_REQ) * rand_f + MIN_REQ;
                std::shared_ptr<void> addr(stream_alloc->alloc(size),
                                           stream_free);
                checker.add(addr.get(), size);
                allocated_ptrs.emplace_back(std::move(addr));
            }
        }

        if (thread_num)
            return;

        // the following only runs on thread 0

        worker_finished[thread_num].store(true);

        for (auto&& i : worker_finished) {
            while (!i.load())
                ;
            if (!failed_msg.empty())
                return;
        }

        dummy_alloc_peak_usage = dummy_alloc->peak_usage();
        checker_peak_usage = checker.peak_usage();
        auto pfill = dummy_alloc->alloc(dummy_alloc->free_size());
        // device memory allocator does not reclaim memory to root allocator
        ASSERT_EQ(0u, dummy_alloc->nr_free());
        ASSERT_NE(nullptr, pfill);

        dev_alloc->print_memory_state();
        // check for memory being moved between streams
        auto size = std::max(stream_alloc->get_free_memory().max,
                             dev_alloc->get_free_memory().max) +
                    10;
        auto addr = stream_alloc->alloc_shared(size);
        checker.add(addr.get(), size);
        allocated_ptrs.emplace_back(std::move(addr));

        dummy_alloc->free(pfill);
    };

    auto worker = [&](size_t thread_num) {
        MGB_TRY { worker_impl(thread_num); }
        MGB_CATCH(std::exception & exc, {
            MGB_LOCK_GUARD(mutex);
            failed_msg =
                    ssprintf("worker %zu failed: %s", thread_num, exc.what());
            mgb_log("%s", failed_msg.c_str());
        });
        worker_finished[thread_num].store(true);
    };

    std::vector<std::thread> threads;
    for (size_t i = 0; i < NR_THREAD; ++i)
        threads.emplace_back(worker, i);

    while (nr_ready_start.load() != NR_THREAD)
        ;
    start_signal.store(true);

    for (auto&& i : threads)
        i.join();

    ASSERT_TRUE(failed_msg.empty()) << failed_msg;

    mgb_log("peak usage ratio: %zu/%zu=%.5f; "
            "backend_nr_alloc: %zu; backend_nr_free: %zu",
            checker_peak_usage, dummy_alloc_peak_usage,
            double(checker_peak_usage) / dummy_alloc_peak_usage,
            dummy_alloc->nr_alloc(), dummy_alloc->nr_free());
    EXPECT_TRUE(dummy_alloc->ever_failed()) << "this fails occasionally";
    ASSERT_GT(dummy_alloc->nr_alloc(), dummy_alloc->nr_free());
    dev_alloc.reset();

    ASSERT_EQ(dummy_alloc->nr_alloc(), dummy_alloc->nr_free());
}

TEST(TestSimpleCachingAlloc, Basic) {
    constexpr size_t TOT = 2048, REQ = 1000;
    static_assert(TOT > REQ * 2, "");
    auto raw_alloc = new DummyAllocator(TOT);
    auto alloc = SimpleCachingAlloc::make(std::unique_ptr<RawAllocator>(raw_alloc));

    auto ptr = alloc->alloc(REQ);
    EXPECT_EQ(TOT - REQ, raw_alloc->free_size());
    EXPECT_EQ(REQ, alloc->get_used_memory());
    EXPECT_EQ(0u, alloc->get_free_memory().tot);

    alloc->free(ptr);
    EXPECT_EQ(0u, raw_alloc->nr_free());
    EXPECT_EQ(REQ, alloc->get_free_memory().tot);

    ptr = alloc->alloc(REQ / 2);
    EXPECT_EQ(1u, raw_alloc->nr_alloc());
    EXPECT_EQ(REQ / 2, alloc->get_used_memory());
    EXPECT_EQ(REQ - REQ / 2, alloc->get_free_memory().tot);

    auto ptr2 = alloc->alloc(REQ / 2);
    EXPECT_EQ(1u, raw_alloc->nr_alloc());
    EXPECT_EQ(REQ / 2 * 2, alloc->get_used_memory());
    EXPECT_EQ(REQ - REQ / 2 * 2, alloc->get_free_memory().tot);
    EXPECT_EQ(REQ / 2, (char*)ptr2 - (char*)ptr);

    alloc->free(ptr);
    EXPECT_EQ(1u, raw_alloc->nr_alloc());
    EXPECT_EQ(REQ / 2, alloc->get_used_memory());
    EXPECT_EQ(REQ - REQ / 2, alloc->get_free_memory().tot);

    ptr = alloc->alloc(REQ);
    EXPECT_EQ(2u, raw_alloc->nr_alloc());
    EXPECT_EQ(TOT - REQ * 2, raw_alloc->free_size());
    EXPECT_EQ(REQ + REQ / 2, alloc->get_used_memory());
    EXPECT_EQ(REQ - REQ / 2, alloc->get_free_memory().tot);

    alloc->free(ptr2);
    ptr2 = alloc->alloc(REQ);
    EXPECT_EQ(2u, raw_alloc->nr_alloc());
    EXPECT_EQ(REQ * 2, alloc->get_used_memory());
    EXPECT_EQ(0u, alloc->get_free_memory().tot);

    alloc->free(ptr);
    alloc->free(ptr2);
    EXPECT_EQ(0u, raw_alloc->nr_free());
};

namespace {
class DevicePolicy {
public:
    virtual void set_device(int device) = 0;
    virtual void get_mem_info(size_t& free, size_t& tot) = 0;
    virtual void raw_dev_malloc(void** ptr, size_t size) = 0;
    virtual void raw_dev_free(void* ptr) = 0;
    virtual ~DevicePolicy() = default;
};

#if MGB_CUDA
class CudaDevicePolicy : public DevicePolicy {
public:
    void set_device(int device) override {
        MGB_CUDA_CHECK(cudaSetDevice(device));
    }
    void get_mem_info(size_t& free, size_t& tot) override {
        MGB_CUDA_CHECK(cudaMemGetInfo(&free, &tot));
    }
    void raw_dev_malloc(void** ptr, size_t size) override {
        MGB_CUDA_CHECK(cudaMalloc(ptr, size));
    }
    void raw_dev_free(void* ptr) override { MGB_CUDA_CHECK(cudaFree(ptr)); }
};
#endif

using Callback = std::function<void()>;
void test_free_mem(CompNode::Locator loc0, CompNode::Locator loc1, DevicePolicy* policy,
                   const Callback& before_run, const Callback& after_run) {
    size_t tot, free;
    policy->set_device(0);
    policy->get_mem_info(free, tot);

    // exception
    auto do_run = [loc0, loc1, policy, free]() {
        void* tmp;
        policy->raw_dev_malloc(&tmp, free / 3);
        auto dev_free = [&](void* ptr) {
            policy->raw_dev_free(ptr);
        };
        auto cn0 = CompNode::load(loc0), cn1 = CompNode::load(loc1);
        std::unique_ptr<void, decltype(dev_free)> tmp_owner{tmp, dev_free};
        auto check_free = [&](const char* msg, size_t expect) {
            auto get = cn0.get_mem_status_bytes().second;
            ASSERT_LE(std::abs(static_cast<intptr_t>(get) -
                               static_cast<intptr_t>(expect)),
                      static_cast<intptr_t>(free) / 4)
                    << ssprintf("%s: get=%.2fMiB expect=%.2fMiB", msg,
                                get / 1024.0 / 1024, expect / 1024.0 / 1024);
        };

        check_free("direct get", free * 2 / 3);
        DeviceTensorStorage tensor{cn0};
        tensor.ensure_size(free / 3).ptr();
        check_free("after dev alloc", free / 3);
        tmp_owner.reset();
        check_free("after outer release", free * 2 / 3);
        tensor = {cn0};
        check_free("after all release", free);

        DeviceTensorStorage tensor1{cn1};
        tensor.ensure_size(free / 6).ptr();
        tensor1.ensure_size(free / 6).ptr();
        check_free("multiple streams", free * 2 / 3);
    };

    before_run();
    MGB_TRY { do_run(); }
    MGB_FINALLY(after_run(););
}

void test_gather_other(CompNode cn0, CompNode cn1) {
    if (cn0.get_mem_status_bytes().second > cn1.get_mem_status_bytes().second) {
        std::swap(cn0, cn1);
    }
    size_t elems = cn0.get_mem_status_bytes().second * 2 / 5 / sizeof(dt_int32);
    auto xv = std::make_shared<DeviceTensorND>(cn0, TensorShape{elems},
                                               dtype::Int32());
    auto graph = ComputingGraph::make();
    auto x = opr::SharedDeviceTensor::make(*graph, xv), x1 = x + 1,
         x2 = opr::MarkDynamicVar::make(x), y = opr::Copy::make(x1, {cn1});
    // x1 must be released (which requires y to finish) before x2 succeeds

    set_priority(x1, -10);
    set_priority(y, -10);
    graph->options().var_sanity_check_first_run = false;
    graph->options().async_exec_level = 0;
    auto func = graph->compile({{x2, {}}, {y, {}}});
    opr::Sleep::sleep(cn1, 0.7);
    func->execute();
}

}  // namespace

#if MGB_CUDA
TEST(TestCudaMemAlloc, GatherOther) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0"), cn1 = CompNode::load("gpu1");
    test_gather_other(cn0, cn1);
}

TEST(TestCudaMemAlloc, FreeMem) {
    // check whether cuda device free mem is correctly impelmented
    REQUIRE_GPU(1);
    CompNode::finalize();
    // same device but different stream
    using Locator = CompNode::Locator;
    auto loc0 = Locator::parse("gpu0"), loc1 = Locator::parse("gpu0:1");
    auto policy = std::make_unique<CudaDevicePolicy>();

    constexpr const char* KEY = "MGB_CUDA_RESERVE_MEMORY";
    auto old_value = getenv(KEY);
    auto reserve = [&]() { setenv(KEY, "1", 1); };
    auto restore = [&]() {
        if (old_value) {
            setenv(KEY, old_value, 1);
        } else {
            unsetenv(KEY);
        }
        CompNode::finalize();
    };
    test_free_mem(loc0, loc1, policy.get(), reserve, restore);
}
#endif  // MGB_CUDA


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
