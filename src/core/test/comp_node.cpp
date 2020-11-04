/**
 * \file src/core/test/comp_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./comp_node_helper.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/utils/comp_node_sync_manager.h"
#include "megbrain/utils/timer.h"
#include "megbrain/system.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/utility.h"

#include <chrono>
#if MGB_HAVE_THREAD
#include <thread>
#endif

using namespace mgb;

TEST(TestCompNode, Parse) {
    using L = CompNode::Locator;
    using D = CompNode::DeviceType;
    auto make_lc = [](D t, int dev, int s) -> L { return {t, dev, {s}}; };

    ASSERT_EQ(L::parse("xpux"), make_lc(D::UNSPEC, -1, 0));
    ASSERT_EQ(L::parse("xpux:23"), make_lc(D::UNSPEC, -1, 23));
    ASSERT_EQ(L::parse("xpu2:23"), make_lc(D::UNSPEC, 2, 23));
    ASSERT_EQ(L::parse("xpu21:23"), make_lc(D::UNSPEC, 21, 23));

    ASSERT_EQ(L::parse("cpux"), make_lc(D::CPU, -1, 0));
    ASSERT_EQ(L::parse("cpux:23"), make_lc(D::CPU, -1, 23));
    ASSERT_EQ(L::parse("cpu2:23"), make_lc(D::CPU, 2, 23));
    ASSERT_EQ(L::parse("cpu21:23"), make_lc(D::CPU, 21, 23));


    ASSERT_EQ(L::parse("rocmx"), make_lc(D::ROCM, -1, 0));
    ASSERT_EQ(L::parse("rocm2"), make_lc(D::ROCM, 2, 0));
    ASSERT_EQ(L::parse("rocm2:3"), make_lc(D::ROCM, 2, 3));
    ASSERT_EQ(L::parse("cambriconx"), make_lc(D::CAMBRICON, -1, 0));
    ASSERT_EQ(L::parse("cambricon2"), make_lc(D::CAMBRICON, 2, 0));
    ASSERT_EQ(L::parse("cambricon2:3"), make_lc(D::CAMBRICON, 2, 3));

    ASSERT_EQ(L::parse("atlasx"), make_lc(D::ATLAS, -1, 0));
    ASSERT_EQ(L::parse("atlas2"), make_lc(D::ATLAS, 2, 0));
    ASSERT_EQ(L::parse("atlas2:3"), make_lc(D::ATLAS, 2, 3));
    ASSERT_EQ(L::parse("xpu"), make_lc(D::UNSPEC, -1, 0));
    ASSERT_EQ(L::parse("xpux"), make_lc(D::UNSPEC, -1, 0));
    ASSERT_EQ(L::parse("xpu23"), make_lc(D::UNSPEC, 23, 0));
    ASSERT_EQ(L::parse("xpu23:1"), make_lc(D::UNSPEC, 23, 1));

    ASSERT_EQ(L::parse("cpu:default"), make_lc(D::CPU, L::DEVICE_CPU_DEFAULT, 0));
    ASSERT_EQ(L::parse("multithread2:0"), make_lc(D::MULTITHREAD, 0, 2));
    ASSERT_EQ(L::parse("multithread1:3"), make_lc(D::MULTITHREAD, 3, 1));
    ASSERT_EQ(L::parse("multithread:default:2"),
              make_lc(D::MULTITHREAD, L::DEVICE_MULTITHREAD_DEFAULT, 2));


    ASSERT_THROW(L::parse("apu"), MegBrainError);
    ASSERT_THROW(L::parse("fpgbx"), MegBrainError);
    ASSERT_THROW(L::parse("cab0"), MegBrainError);
    ASSERT_THROW(L::parse("cpu"), MegBrainError);
    ASSERT_THROW(L::parse("cpu-1"), MegBrainError);
    ASSERT_THROW(L::parse("cpu0:"), MegBrainError);
    ASSERT_THROW(L::parse("cpu0:x"), MegBrainError);
    ASSERT_THROW(L::parse("cpu2:23x"), MegBrainError);
    ASSERT_THROW(L::parse("rcom0"), MegBrainError);
    ASSERT_THROW(L::parse("cmabricon0"), MegBrainError);
    ASSERT_THROW(L::parse("atlast0"), MegBrainError);
    ASSERT_THROW(L::parse("multithread"), MegBrainError);
    ASSERT_THROW(L::parse("multithread1:"), MegBrainError);
    ASSERT_THROW(L::parse("multithread1:default"), MegBrainError);
    ASSERT_THROW(L::parse("multithread1:default:0"), MegBrainError);
}

TEST(TestCompNode, SetDefaultDev) {
    REQUIRE_GPU(3);

    CompNode::finalize();
    using L = CompNode::Locator;
    auto orig_dt = L::parse("xpu").to_physical(),
         orig_gpu = L::parse("gpux").to_physical();
    constexpr auto CUDA = CompNode::DeviceType::CUDA;
    L::set_unspec_device_type(CUDA);
    L::set_device_map(CUDA, -1, 2);
    auto run = []() {
        ASSERT_EQ(CompNode::load("xpu").locator(), L::parse("gpu2"));
    };

    MGB_TRY {
        run();
    } MGB_FINALLY({
        L::set_unspec_device_type(orig_dt.type);
        L::set_device_map(CUDA, -1, orig_gpu.device);
    });
    CompNode::finalize();
}

TEST(TestCompNode, Load) {
    auto cn0 = CompNode::load("xpux"),
         cn1 = CompNode::load("cpux");
    ASSERT_EQ(CompNode::DeviceType::UNSPEC, cn0.locator_logical().type);
    ASSERT_EQ(CompNode::DeviceType::CPU, cn1.locator_logical().type);
    ASSERT_EQ(CompNode::load("cpux"), cn1);
    ASSERT_EQ(CompNode::load("xpux"), cn0);
    auto cnp = CompNode::load("cpu1"), cnq = CompNode::load("cpu2");
    ASSERT_EQ(CompNode::load("cpu1"), cnp);
    ASSERT_EQ(CompNode::load("cpu2"), cnq);
#if MGB_HAVE_THREAD
    ASSERT_NE(cnp, cnq);
#else
    ASSERT_EQ(cnp, cnq);
#endif

#if MGB_HAVE_THREAD
    auto cn_multi_thread0 = CompNode::load("multithread2:0");
    auto cn_multi_thread1 = CompNode::load("multithread2:1");
    ASSERT_EQ(CompNode::load("multithread2:0"), cn_multi_thread0);
    ASSERT_EQ(CompNode::load("multithread2:1"), cn_multi_thread1);
    ASSERT_NE(CompNode::load("multithread4:0"), cn_multi_thread0);
    ASSERT_NE(CompNode::load("multithread4:1"), cn_multi_thread1);

    auto cn_multi_default0 = CompNode::load("multithread:default:2");
    auto cn_multi_default1 = CompNode::load("multithread:default:4");
    ASSERT_EQ(CompNode::load("multithread:default:2"), cn_multi_default0);
    ASSERT_EQ(CompNode::load("multithread:default:4"), cn_multi_default1);
    ASSERT_NE(cn_multi_thread0, cn_multi_default1);
#endif

    ASSERT_EQ(CompNode::load("cpu1"), cnp);
    ASSERT_EQ(CompNode::load("cpu2"), cnq);
    if (check_gpu_available(2)) {
        auto cn2 = CompNode::load("gpux"),
             cn3 = CompNode::load("gpu1");
        ASSERT_EQ(CompNode::DeviceType::CUDA, cn2.locator_logical().type);
        ASSERT_NE(cn2, cn3);
        ASSERT_EQ(CompNode::load("gpux"), cn2);
        ASSERT_EQ(CompNode::load("gpu1"), cn3);
    }

#if MGB_ATLAS
    auto atlas0 = CompNode::load("atlas0");
    auto atlas1 = CompNode::load("atlas1");
    ASSERT_NE(atlas0, atlas1);
#endif

}

TEST(TestCompNode, FreeAfterFinalize) {
    CompNode::finalize();
    for (size_t i = 0; i < CompNode::NR_DEVICE_TYPE; ++i) {
        auto type = static_cast<CompNode::DeviceType>(i);
        if (!check_device_type_avaiable(type) ||
            !CompNode::get_device_count(type))
            continue;
        auto cn = CompNode::load(CompNode::Locator{type, -1, {0}});
        auto ptr = cn.alloc_device(123);
        CompNode::finalize();
        cn.free_device(ptr);
    }
}

TEST(TestCompNode, CPUDispatchSync) {
    REQUIRE_THREAD();
    constexpr int LOOP = 160, tot_threads = 8;
    std::atomic_int started_threads{0};
    auto worker = [&](int *shared_cnt, CompNode dest) {
        int nr_call = 0;
        RNGxorshf rng{next_rand_seed()};
        auto func = [&rng, &nr_call, shared_cnt]() {
            ++ nr_call;
            ++ *shared_cnt;
            int volatile cnt = 0;
            while (rng() % 20)
                ++ cnt;
        };
        auto &&env = CompNodeEnv::from_comp_node(dest).cpu_env();
        ++ started_threads;
        while (started_threads.load() != tot_threads);
        for (int i = 0; i < LOOP; ++ i) {
            env.dispatch(func);
            dest.sync();
            ASSERT_EQ(i + 1, nr_call);
        }
    };
    auto cn0 = CompNode::load("cpu0"), cn1 = CompNode::load("cpu1");
    int cnt0 = 0, cnt1 = 0;
    std::vector<std::thread> wk_threads;
    for (int i = 0; i < tot_threads / 2; ++ i) {
        wk_threads.emplace_back(worker, &cnt0, cn0);
        wk_threads.emplace_back(worker, &cnt1, cn1);
    }

    for (auto &&i: wk_threads)
        i.join();

    ASSERT_EQ(LOOP * tot_threads / 2, cnt0);
    ASSERT_EQ(LOOP * tot_threads / 2, cnt1);
}

TEST(TestCompNodeCPU, CoreAffinity) {
    REQUIRE_THREAD();
    std::vector<size_t> data_v(2, 0);
    size_t data0, data1 = 0;
    auto empty_task = []() {};
    auto cn0 = CompNode::load("cpu:default"), cn1 = CompNode::load("cpu0"),
         cn2 = CompNode::load("multithread2:0");
    auto binding0 = [&](size_t) { data0 = 10; };
    CompNodeEnv::from_comp_node(cn0).cpu_env().set_affinity(binding0);
    CompNodeEnv::from_comp_node(cn0).cpu_env().dispatch(empty_task);
    cn0.sync();

    auto binding1 = [&](size_t ) { data1 = 20; };
    CompNodeEnv::from_comp_node(cn1).cpu_env().set_affinity(binding1);
    CompNodeEnv::from_comp_node(cn1).cpu_env().dispatch(empty_task);
    cn1.sync();

    auto binding2 = [&](size_t thread_id) { data_v[thread_id] = 30; };
    auto temp_task = [](size_t, size_t) {};
    CompNodeEnv::from_comp_node(cn2).cpu_env().set_affinity(binding2);
    CompNodeEnv::from_comp_node(cn2).cpu_env().dispatch(temp_task, 40u);
    cn2.sync();
    ASSERT_EQ(data0, static_cast<size_t>(10));
    ASSERT_EQ(data1, static_cast<size_t>(20));
    ASSERT_EQ(data_v[0], static_cast<size_t>(30));
    ASSERT_EQ(data_v[1], static_cast<size_t>(30));
}

TEST(TestCompNode, CPU_MULTI_THREAD) {
    REQUIRE_THREAD();
    std::vector<int> source(100), dst0(100), dst1(100);
    for (int i = 0; i < 100; i++) {
        source[i] = i;
        dst0[i] = 0;
        dst1[i] = 0;
    }
    size_t total_task = 20;
    auto worker = [&](std::vector<int>& dst, CompNode dest) {
        auto func = [&](size_t index, size_t) {
            size_t sub_task = 100 / total_task;
            for (size_t i = index * sub_task; i < (index + 1) * sub_task; i++) {
                int sum = 0;
                for (size_t j = 0; j < i; j++) {
                    sum += source[j];
                }
                dst[i] = sum;
            }
        };
        auto&& env = CompNodeEnv::from_comp_node(dest).cpu_env();
        env.dispatch(std::move(func), total_task);
        dest.sync();
    };

    for (auto&& str : std::vector<std::string>{
                 "multithread2:0", "multithread4:0", "multithread:default:4"}) {
        auto cn0 = CompNode::load("cpu0"), cn1 = CompNode::load(str);
        std::thread wk_thread0{std::ref(worker), std::ref(dst0), std::ref(cn0)};
        std::thread wk_thread1{std::ref(worker), std::ref(dst1), std::ref(cn1)};

        wk_thread0.join();
        wk_thread1.join();

        for (int i = 0; i < 100; i++) {
            ASSERT_EQ(dst0[i], dst1[i]);
        }
    }
}

TEST(TestCompNodeCuda, MemNode) {
    REQUIRE_GPU(2);

    auto cn00 = CompNode::load("gpu0"),
         cn1 = CompNode::load("gpu1"),
         cn01 = CompNode::load("gpu0:1");
    ASSERT_EQ(cn00, CompNode::load("gpu0"));
    ASSERT_EQ(cn00.mem_node(), cn01.mem_node());
    ASSERT_NE(cn00.mem_node(), cn1.mem_node());
}

TEST(TestCompNodeCuda, Uid) {
    REQUIRE_GPU(2);

    auto cn00 = CompNode::load("gpu0"),
         cn1 = CompNode::load("gpu1"),
         cn01 = CompNode::load("gpu0:0"),
         cn02 = CompNode::load("gpu0:2");
    ASSERT_EQ(cn00, CompNode::load("gpu0"));
    ASSERT_EQ(cn00.get_uid(), cn01.get_uid());
    ASSERT_NE(cn00.get_uid(), cn02.get_uid());
    ASSERT_NE(cn00.get_uid(), cn1.get_uid());
}

TEST(TestCompNodeCuda, set_prealloc_config) {
    CompNode::set_prealloc_config(
        1024, 1024, 256 * 1024 * 1024,
        4, CompNode::DeviceType::CUDA);
}

#if MGB_ROCM
TEST(TestCompNodeROCm, MemNode) {
    REQUIRE_AMD_GPU(2);

    auto cn00 = CompNode::load("rocm0"),
         cn1 = CompNode::load("rocm1"),
         cn01 = CompNode::load("rocm0:1");
    ASSERT_EQ(cn00, CompNode::load("rocm0"));
    ASSERT_EQ(cn00.mem_node(), cn01.mem_node());
    ASSERT_NE(cn00.mem_node(), cn1.mem_node());
}
#endif

#if MGB_CAMBRICON
TEST(TestCompNodeCambricon, MemNode) {
    REQUIRE_CAMBRICON_DEVICE(2);
    auto cn00 = CompNode::load("cambricon0"),
         cn1 = CompNode::load("cambricon1"),
         cn01 = CompNode::load("cambricon0:1");
    ASSERT_EQ(cn00, CompNode::load("cambricon0"));
    ASSERT_EQ(cn00.mem_node(), cn01.mem_node());
    ASSERT_NE(cn00.mem_node(), cn1.mem_node());
}
#endif

#if MGB_ATLAS
TEST(TestCompNodeAtlas, MemNode) {
    auto cn00 = CompNode::load("atlas0"),
         cn1 = CompNode::load("atlas1"),
         cn01 = CompNode::load("atlas0:1");
    ASSERT_EQ(cn00, CompNode::load("atlas0"));
    ASSERT_EQ(cn00.mem_node(), cn01.mem_node());
    ASSERT_NE(cn00.mem_node(), cn1.mem_node());
}
#endif


TEST(TestCompNodeCPU, PhysicalDispatch) {
    constexpr int ID = 0x2a6453e0;
    using L = CompNode::Locator;
    constexpr auto DT = CompNode::DeviceType::CPU;
    L::set_device_map(DT, ID, 0);
    L::set_device_map(DT, ID + 1, 0);
    L::set_device_map(DT, ID + 2, 1);
    auto cn0 = CompNode::load({DT, ID, {0}}),
         cn1 = CompNode::load({DT, ID + 1, {0}}),
         cn2 = CompNode::load({DT, ID + 2, {0}});
#if MGB_HAVE_THREAD
    ASSERT_NE(cn0, cn1);
#else
    ASSERT_EQ(cn0, cn1);
#endif
    std::vector<std::thread::id> tids;
    std::mutex tids_mtx;
    auto get_tid = [&]() {
        MGB_LOCK_GUARD(tids_mtx);
        tids.push_back(std::this_thread::get_id());
    };
    CompNodeEnv::from_comp_node(cn0).cpu_env().dispatch(get_tid);
    CompNodeEnv::from_comp_node(cn1).cpu_env().dispatch(get_tid);
    CompNodeEnv::from_comp_node(cn2).cpu_env().dispatch(get_tid);
    CompNode::sync_all();
    std::unordered_set<std::thread::id> uniq_tids(tids.begin(), tids.end());
    ASSERT_EQ(3u, tids.size());
#if MGB_HAVE_THREAD
    ASSERT_EQ(2u, uniq_tids.size());
#else
    ASSERT_EQ(1u, uniq_tids.size());
#endif
}

TEST(TestCompNodeCPU, EventWait) {
    REQUIRE_THREAD();
    std::atomic_bool start = ATOMIC_VAR_INIT(false);
    auto cn0 = CompNode::load("cpu0"),
         cn1 = CompNode::load("cpu1");
    auto task0 = [&]() {
        while (!start)
            std::this_thread::yield();
    };
    auto event = cn0.create_event();
    CompNodeEnv::from_comp_node(cn0).cpu_env().dispatch(task0);
    event->record();
    cn1.device_wait_event(*event);

    bool succ = false;
    auto task1 = [&]() {
        succ = start;
    };
    CompNodeEnv::from_comp_node(cn1).cpu_env().dispatch(task1);

    using namespace std::literals;
    std::this_thread::sleep_for(50ms);
    ASSERT_FALSE(succ);
    start = true;
    CompNode::sync_all();
    ASSERT_TRUE(succ);
}

TEST(TestCompNodeCPU, EventRecOverwrite) {
    REQUIRE_THREAD();
    auto cn = CompNode::load("cpu0");
    auto dispatcher = CompNodeEnv::from_comp_node(cn).
        cpu_env().dispatcher.get();
    auto dispatch = [&](MegcoreCPUDispatcher::Task &&t) {
        dispatcher->dispatch(std::move(t));
    };
    auto ev = cn.create_event();
    auto wait_atomic = [](std::atomic_bool *var) {
        while(!var->load())
            std::this_thread::yield();
    };
    auto set_atomic = [](std::atomic_bool *var) {
        var->store(true);
    };

    std::atomic_bool
        s0 = ATOMIC_VAR_INIT(false),
        s1 = ATOMIC_VAR_INIT(false),
        t0 = ATOMIC_VAR_INIT(false),
        t1 = ATOMIC_VAR_INIT(false),
        t2 = ATOMIC_VAR_INIT(false);

    dispatch(std::bind(set_atomic, &t0));
    dispatch(std::bind(wait_atomic, &s0));
    ev->record();
    dispatch(std::bind(set_atomic, &t1));

    dispatch(std::bind(wait_atomic, &s1));
    ev->record();
    dispatch(std::bind(set_atomic, &t2));

    wait_atomic(&t0);
    ASSERT_FALSE(ev->finished());
    set_atomic(&s0);
    wait_atomic(&t1);
    ASSERT_FALSE(ev->finished());
    set_atomic(&s1);
    wait_atomic(&t2);
    ASSERT_TRUE(ev->finished());
}

namespace {
void test_peer_copy_from_device(const char* comp_node) {
    REQUIRE_THREAD();
    auto cn_gpu = CompNode::load(comp_node);
    auto cn_cpu = CompNode::load("cpux");

    HostTensorGenerator<> gen;
    auto a = gen({20, 3, 112, 112});
    auto b = gen({20, 3, 112, 112});
    auto c = gen({20, 3, 112, 112});
    DeviceTensorND dev_a{cn_gpu}, dev_b{cn_cpu}, dev_c{cn_gpu};
    dev_a.copy_from(*a).sync();
    dev_b.copy_from(*b).sync();
    dev_c.copy_from(*c).sync();

    auto wait_event = cn_gpu.create_event();

    opr::Sleep::sleep(cn_gpu, 0.1);
    dev_a.copy_from(dev_c);
    wait_event->record();

    cn_cpu.device_wait_event(*wait_event);
    dev_b.copy_from(dev_a);

    dev_b.sync();

    HostTensorND result;
    result.copy_from(dev_b);

    CompNode::sync_all();

    MGB_ASSERT_TENSOR_EQ(result, *c);
}
}

TEST(TestCompNodeCPU, PeerCopyFromCUDA) {
    REQUIRE_GPU(1);
    test_peer_copy_from_device("gpux");
}

TEST(TestCompNodeCPU, PeerCopyFromROCm) {
    REQUIRE_AMD_GPU(1);
    test_peer_copy_from_device("rocmx");
}

#if MGB_CAMBRICON
TEST(TestCompNodeCPU, PeerCopyFromCambricon) {
    REQUIRE_CAMBRICON_DEVICE(1);
    REQUIRE_THREAD();
    auto cn_gpu = CompNode::load("cambriconx");
    auto cn_cpu = CompNode::load("cpux");

    HostTensorGenerator<> gen;
    auto a = gen({20, 3, 112, 112});
    auto b = gen({20, 3, 112, 112});
    auto c = gen({20, 3, 112, 112});
    DeviceTensorND dev_a{cn_gpu}, dev_b{cn_cpu}, dev_c{cn_gpu};
    dev_a.copy_from(*a).sync();
    dev_b.copy_from(*b).sync();
    dev_c.copy_from(*c).sync();

    auto wait_event = cn_gpu.create_event();

    dev_a.copy_from(dev_c);
    wait_event->record();

    cn_cpu.device_wait_event(*wait_event);
    dev_b.copy_from(dev_a);

    dev_b.sync();

    HostTensorND result;
    result.copy_from(dev_b);

    CompNode::sync_all();

    MGB_ASSERT_TENSOR_EQ(result, *c);
}
#endif

TEST(TestCompNodeSyncManager, HostWait) {
    REQUIRE_THREAD();
    CompNodeSyncManager mgr(CompNode::load("xpu0"));

    auto run_set = [&]() {
        using namespace std::literals;
        std::this_thread::sleep_for(200ms);
        mgr.set_ready();
        mgb_log_debug("set_ready() called");
    };

    for (int run = 0; run < 2; ++ run) {
        std::thread th_run_set(run_set);

        RealTimer timer;
        mgr.clear_waiter_record();
        ASSERT_THROW(mgr.busy_wait_set_ready(), MegBrainError);

        mgr.add_waiter_record(false);
        mgr.add_waiter_record(false);
        mgr.busy_wait_set_ready();
        EXPECT_GE(timer.get_secs(), 0.1);
        timer.reset();
        mgr.busy_wait_set_ready();
        EXPECT_LE(timer.get_secs(), 0.001);

        th_run_set.join();
    }
}

TEST(TestCompNodeSyncManager, DeviceWait) {
    REQUIRE_THREAD();
    auto cns = load_multiple_xpus(3);
    auto cn0 = cns[0], cn1 = cns[1], cn2 = cns[2];
    CompNodeSyncManager mgr(cn0);

    using Event = CompNode::Event;
    auto ev_cn1 = cn1.create_event(),
         ev_cn2_begin = cn2.create_event(Event::NEED_TIMER),
         ev_cn2_end = cn2.create_event(Event::NEED_TIMER);

    for (int run = 0; run < 2; ++ run) {
        RealTimer timer;
        mgr.clear_waiter_record();
        ASSERT_THROW(mgr.busy_wait_set_ready_and_get_event(), MegBrainError);
        mgr.add_waiter_record(true);
        mgr.add_waiter_record(true);
        opr::Sleep::sleep(cn0, 0.13);
        mgr.set_ready();
        ev_cn2_begin->record();
        cn1.device_wait_event(mgr.busy_wait_set_ready_and_get_event());
        cn2.device_wait_event(mgr.busy_wait_set_ready_and_get_event());
        ev_cn1->record();
        ev_cn2_end->record();
        EXPECT_LE(timer.get_secs(), 0.05);

        ev_cn1->host_wait();
        EXPECT_GE(timer.get_secs(), 0.1);
        ev_cn2_end->host_wait();
        auto ev2_t = ev_cn2_begin->elapsed_time_until(*ev_cn2_end);
        EXPECT_GE(ev2_t, 0.1);
    }
}

TEST(TestCompNodeSyncManager, DeviceWaitCross) {
    REQUIRE_THREAD();
    auto cn0 = CompNode::load("xpu0:0"), cn1 = CompNode::load("xpu0:1");
    auto ev_cn0 = cn0.create_event(),
         ev_cn1 = cn1.create_event();

    RealTimer timer;

    // cross wait like deadlock, but guaranteed to work due to good timing
    ev_cn0->record();
    cn1.device_wait_event(*ev_cn0);
    ev_cn1->record();
    opr::Sleep::sleep(cn0, 0.1);
    cn0.device_wait_event(*ev_cn1);
    ev_cn0->record();
    cn1.device_wait_event(*ev_cn0);

    cn0.sync();
    cn1.sync();
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto used = timer.get_secs();
    if (used <= 0.1 || used >= 0.2) {
        mgb_log_warn("expect time between [%f, %f], got %f", 0.1, 0.2, used);
    }
}

#if !MGB_HAVE_THREAD
TEST(TestCompNodeSyncManager, DeviceWaitWithoutThread) {
    auto cn = CompNode::load("cpu:default");
    CompNodeSyncManager mgr(cn);
    mgr.add_waiter_record(true);
    ASSERT_ANY_THROW(mgr.busy_wait_set_ready());
    mgr.set_ready();
    EXPECT_TRUE(mgr.busy_wait_set_ready_and_get_event().finished());
}
#endif

TEST(TestCompNode, MultipleLoad) {
    auto run = [](CompNode cn) {
        HostTensorND a(cn, {23}, dtype::Int32{}), b;
        auto pa = a.ptr<int>();
        for (int i = 0; i < 23; ++i) {
            pa[i] = i;
        }
        DeviceTensorND tmp;
        tmp.copy_from(a);
        b.copy_from(tmp).sync();
        auto pb = b.ptr<int>();
        for (int i = 0; i < 23; ++i) {
            ASSERT_EQ(i, pb[i]);
        }
        CompNode::finalize();
    };
    for (size_t i = 1; i < CompNode::NR_DEVICE_TYPE; ++i) {
        auto dt = static_cast<CompNode::DeviceType>(i);
        if (!check_device_type_avaiable(dt))
            continue;
        if (CompNode::get_device_count(dt)) {
            auto cn = CompNode::load({dt, 0, {0}});
            mgb_log("comp node %s is available", cn.to_string().c_str());
            run(cn);
            cn = CompNode::load({dt, 0, {0}});
            run(cn);
        }
    }
}

#if MGB_CAMBRICON
TEST(TestCompNodeCambricon, D2DCopy) {
auto run = [](CompNode cn) {
        constexpr size_t size = 100 * 1024 * 1024;
        HostTensorND a(cn, {size}, dtype::Int32{}), b;
        auto pa = a.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            pa[i] = i;
        }
        DeviceTensorND tmp, tmp1;
        tmp.copy_from(a);
        tmp1.copy_from(tmp);
        b.copy_from(tmp1).sync();
        auto pb = b.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(static_cast<int>(i), pb[i]);
        }
        CompNode::finalize();
    };
    REQUIRE_CAMBRICON_DEVICE(1);
    auto cn = CompNode::load("cambricon0");
    run(cn);
    cn = CompNode::load("cambricon1");
    run(cn);
}

// peer copy for cambricon between different devices is not correct now, so
// disable this testcase
#if 0
TEST(TestCompNodeCambricon, P2PCopy) {
    auto run_raw = []() {
        int v0 = 0, v1 = 1;
        cnrtDev_t dev0, dev1;
        MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev0, 0));
        MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev1, 1));
        int *dp0, *dp1;
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev0));
        MGB_CNRT_CHECK(cnrtMalloc((void**)(&dp0), sizeof(int)));
        MGB_CNRT_CHECK(
                cnrtMemcpy(dp0, &v0, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev1));
        MGB_CNRT_CHECK(cnrtMalloc((void**)(&dp1), sizeof(int)));
        MGB_CNRT_CHECK(
                cnrtMemcpy(dp1, &v1, sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV));
        unsigned int can = 0;
        MGB_CNRT_CHECK(cnrtGetPeerAccessibility(&can, 0, 1));
        printf("can = %s\n", can ? "TRUE" : "FALSE");
        if (can) {
            MGB_CNRT_CHECK(cnrtMemcpyPeer(dp1, 1, dp0, 0, sizeof(int)));
            int get;
            MGB_CNRT_CHECK(cnrtMemcpy(&get, dp1, sizeof(int),
                                      CNRT_MEM_TRANS_DIR_DEV2HOST));
            ASSERT_EQ(0, get);
        }
    };
    auto run = [](CompNode cn0, CompNode cn1) {
        constexpr size_t size = 100;
        HostTensorND a(cn0, {size}, dtype::Int32{}), b;
        auto pa = a.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            pa[i] = i;
        }
        DeviceTensorND tmp(cn0, {size}, dtype::Int32{}),
                tmp1(cn1, {size}, dtype::Int32{});
        tmp.copy_from(a);
        tmp1.copy_from(tmp);
        b.copy_from(tmp1).sync();
        auto pb = b.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(static_cast<int>(i), pb[i]);
        }
        CompNode::finalize();
    };
    REQUIRE_CAMBRICON_DEVICE(2);
    auto cn0 = CompNode::load("cambricon0"), cn1 = CompNode::load("cambricon1");
    run_raw();
    run(cn0, cn1);
}
#endif
#endif // MGB_CAMBRICON

#if MGB_ATLAS

TEST(TestCompNodeAtlas, D2DCopy) {
    auto run = [](CompNode cn) {
        constexpr size_t size = 10 * 1024 * 1024;
        HostTensorND a(cn, {size}, dtype::Int32{}), b;
        auto pa = a.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            pa[i] = i;
        }
        DeviceTensorND tmp, tmp1;
        tmp.copy_from(a);
        tmp1.copy_from(tmp);
        b.copy_from(tmp1).sync();
        auto pb = b.ptr<int>();
        for (size_t i = 0; i < size; ++i) {
            ASSERT_EQ(static_cast<int>(i), pb[i]);
        }
        CompNode::finalize();
    };
    auto cn = CompNode::load("atlas0");
    run(cn);
}
#endif


namespace {
class CompNodeDepedentObjectInst final : public CompNodeDepedentObject {
    int *m_dst, *m_timer;

    std::shared_ptr<void> on_comp_node_finalize() override {
        EXPECT_EQ(0, *m_dst);
        *m_dst = ++*m_timer;
        return {};
    }

public:
    CompNodeDepedentObjectInst(int* dst, int* timer)
            : m_dst{dst}, m_timer{timer} {}
    void chk() { check_not_finalized(); }
};
}  // anonymous namespace

TEST(TestCompNode, DepedentObjectList) {
    CompNode::finalize();
    for (int i = 0; i < 5; ++i) {
        // loop multiple times so memory problems can be easier exposed
        int ts[4] = {0}, timer = 0;
        auto make = [&](int i) {
            return std::make_unique<CompNodeDepedentObjectInst>(ts + i, &timer);
        };
        auto i0 = make(0), i1 = make(1), i2 = make(2), i3 = make(3);
        ASSERT_NO_THROW(i0->chk());
        ASSERT_NO_THROW(i1->chk());
        i1.reset();
        comp_node_detail::DepedentObjList::invoke_callback_and_clean();
        ASSERT_EQ(1, ts[3]);
        ASSERT_EQ(2, ts[2]);
        ASSERT_EQ(0, ts[1]);
        ASSERT_EQ(3, ts[0]);
        ASSERT_THROW(i0->chk(), InternalError);
    }
}

namespace {
template <typename tag>
class TestCPUCompSeqRec : public ::testing::Test {};
TYPED_TEST_CASE(TestCPUCompSeqRec, comp_node_test::seq_rec::test_types);
TYPED_TEST(TestCPUCompSeqRec, run) {
    comp_node_test::seq_rec::run<TypeParam>(CompNode::load("cpux"));
}
TYPED_TEST(TestCPUCompSeqRec, run_default_cpu) {
    comp_node_test::seq_rec::run<TypeParam>(CompNode::load("cpu:default"));
}
TYPED_TEST(TestCPUCompSeqRec, run_multi_thread) {
    auto cn = CompNode::load("multithread4:0");
    comp_node_test::seq_rec::run<TypeParam>(cn);
}

TYPED_TEST(TestCPUCompSeqRec, run_multi_thread_default) {
    auto cn = CompNode::load("multithread:default:4");
    comp_node_test::seq_rec::run<TypeParam>(cn);
}
}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
