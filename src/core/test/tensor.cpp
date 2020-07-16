/**
 * \file src/core/test/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/tensor.h"
#include "megbrain/opr/utility.h"
#include "megbrain/utils/timer.h"
#include "megbrain/utils/debug.h"
#include "megbrain/exception.h"
#include "megdnn/tensor_format.h"

#include <cmath>

using namespace mgb;

constexpr double ASYNC_SLEEP_TIME = 0.15,
          ASYNC_MAX_ISSUE_TIME = 0.07;

namespace {

template<class Src, class Dst>
void run_noncontig_test() {
    // use a relatively large size so synchronization problems can be detected
    constexpr size_t S0 = 200, S1 = 500;
    HostTensorND hv_init{CompNode::load("xpu0"), dtype::Float32()};
    hv_init.resize({S0, S1});
    for (size_t i = 0; i < S0 * S1; ++ i)
        hv_init.ptr<float>()[i] = i;

    Src src;
    src.copy_from(hv_init);

    bool failed = false;
    auto check = [&](size_t begin, size_t end) {
        ASSERT_FALSE(failed);
        failed = true;

        Src src_sub;
        Dst dst;
        src_sub = src.sub(Slice(begin, end).apply(src.layout(), 1));
        dst.copy_from(src_sub).sync();

        HostTensorND rst;
        rst.copy_from(dst).sync();

        auto ptr = rst.ptr<float>();
        for (size_t i = 0; i < S0; ++ i)
            for (size_t j = begin; j < end; ++ j) {
                ASSERT_EQ(float(i * S1 + j), *ptr);
                ++ ptr;
            }

        HostTensorND hv_zero{hv_init.comp_node(), dtype::Float32()};
        hv_zero.resize({S0, end - begin});
        memset(hv_zero.ptr<float>(), 0, hv_zero.layout().span().dist_byte());
        Dst dst_zero;
        dst_zero.copy_from(hv_zero);
        src_sub.copy_from_fixlayout(dst_zero);
        HostTensorND src_hv;
        src_hv.copy_from(src).sync();
        ptr = src_hv.ptr<float>();
        for (size_t i = 0; i < S0; ++ i)
            for (size_t j = begin; j < end; ++ j) {
                ASSERT_EQ(0.f, ptr[i * S1 + j]);
            }

        src_sub.copy_from_fixlayout(dst).sync();

        failed = false;
    };

    check(0, 1);
    check(S1 - 1, S1);
    check(0, S1 - 1);
    check(1, S1);
    check(12, 21);
}
} // anonymous namespace

TEST(TestTensorStorage, InvalidAlloc) {
    {
        TensorStorage<HostTensorStorageTrait> storage;
        EXPECT_THROW(storage.ensure_size(100), MegBrainError);
    }
    {
        TensorStorage<DeviceTensorStorageTrait> storage;
        EXPECT_THROW(storage.ensure_size(100), MegBrainError);
    }
}

TEST(TestTensorStorage, CopyFromFixLayoutImage2DPack4TensorFormat) {
    CompNode cn = CompNode::load("xpu0");
    HostTensorND dst(
            cn, TensorLayout(TensorShape{1, 1, 1, 1, 4}, dtype::Float32{},
                             megdnn::DefaultTensorFormat::make()));
    HostTensorGenerator<> gen;
    auto src_default = gen({1, 1, 1, 1, 4});
    HostTensorND src(
            cn,
            TensorLayout(TensorShape{1, 1, 1, 1, 4}, dtype::Float32{},
                         megdnn::Image2DPack4TensorFormat::make_raw(2, 64)));

    EXPECT_NO_THROW(src.copy_from_fixlayout(*src_default).sync());
    EXPECT_NO_THROW(dst.copy_from_fixlayout(src).sync());
    MGB_ASSERT_TENSOR_EQ(src, dst);
}

TEST(TestTensorStorage, H2HCopy) {
    HostTensorGenerator<> gen;
    HostTensorND t1;
    auto t0 = gen({123, 456});
    t1.copy_from(*t0);
    MGB_ASSERT_TENSOR_EQ(*t0, t1);
}

TEST(TestTensorStorage, H2DCopy) {
    HostTensorGenerator<> gen;
    auto t0 = gen({123, 456});
    DeviceTensorND t1;
    t1.copy_from(*t0);
    HostTensorND t2;
    t2.copy_from(t1).sync();
    MGB_ASSERT_TENSOR_EQ(*t0, t2);
}

TEST(TestTensorStorage, D2DGPU2DefaultCPU) {
    REQUIRE_GPU(1);

    HostTensorGenerator<> gen;
    HostTensorND host_get;
    auto host_val = gen({123});
    auto cn0 = CompNode::load("gpu0");
    DeviceTensorND t0{cn0}, t1{CompNode::default_cpu()};
    opr::Sleep::sleep(cn0, 0.1);
    t0.copy_from(*host_val);
    t1.copy_from(t0);
    host_get.copy_from(t1);
    MGB_ASSERT_TENSOR_EQ(*host_val, host_get);
}

TEST(TestTensorStorage, D2DCopyNoSync) {
    auto cns = load_multiple_xpus(2);
    HostTensorND t0(cns[0], {1}), t3(cns[1], {1});
    DeviceTensorND t1(cns[0]), t2(cns[1]);
    t0.ptr<float>()[0] = 1;
    t3.ptr<float>()[0] = -1;

    t1.copy_from(t3).sync();
    t2.copy_from(t3).sync();

    RealTimer timer;
    opr::Sleep::sleep(t1.comp_node(), ASYNC_SLEEP_TIME);
    t1.copy_from(t0);
    t2.copy_from(t1);
    t3.copy_from(t2);
    // sleep kernel in cuda is easily affected by the frequency change of GPU,
    // so we just print warn log instead assert. more refer to
    // XPU-226
    auto use_time = timer.get_secs();
    if (use_time >= ASYNC_MAX_ISSUE_TIME) {
        mgb_log_warn("expect time [%f < %f], got %f", use_time,
                     ASYNC_MAX_ISSUE_TIME, use_time);
    }
    t1.sync();
    use_time = timer.get_secs();
    if (use_time <= ASYNC_SLEEP_TIME) {
        mgb_log_warn("expect time [%f > %f], got %f", use_time,
                     ASYNC_MAX_ISSUE_TIME, use_time);
    }
    ASSERT_GT(fabs(t3.sync().ptr<float>()[0] - t0.ptr<float>()[0]), 0.1);
}

TEST(TestTensorStorage, TensorSub) {
    HostTensorND t0(CompNode::load("xpu0"), {123, 456});
    auto t0_sub = t0[{{0, 5}, {1, 9}}];
    ASSERT_EQ(TensorShape({5, 8}), t0_sub.shape());
}

TEST(TestTensorStorage, D2DCopyNonCont) {
    auto cns = load_multiple_xpus(2);
    constexpr size_t S0 = 12, S1 = 8, S2 = 9;
    auto cn0 = cns[0], cn1 = cns[1];
    auto event = cn0.create_event();
    HostTensorND hv(cn0, {S0, S1, S2});
    for (size_t i = 0, it = hv.layout().total_nr_elems(); i < it; i ++)
        hv.ptr<float>()[i] = i;
    DeviceTensorND dv, dv_sub0(cn1), dv_sub1;
    dv.copy_from(hv);
    event->record();
    cn1.device_wait_event(*event);

    dv_sub0.copy_from(dv[{{}, {2, 4}}]);
    dv_sub1.copy_from(dv[{{None, None, 4}, {None, None, 2}, {None, None, 3}}]);

    HostTensorND hv_sub0, hv_sub1;
    hv_sub0.copy_from(dv_sub0);
    hv_sub1.copy_from(dv_sub1);

    auto idx = [](size_t i, size_t j, size_t k) {
        return i * S1 * S2 + j * S2 + k;
    };

    {
        auto ptr = hv_sub0.sync().ptr<float>();
        ASSERT_EQ(TensorShape({S0, 2, S2}), hv_sub0.shape());
        for (size_t i = 0; i < S0; i ++)
            for (size_t j = 0; j < 2; j ++)
                for (size_t k = 0; k < S2; k ++) {
                    MGB_ASSERT_FLOAT_EQ(idx(i, j + 2, k), *(ptr ++)) <<
                        ssprintf("sub0: failed at (%zu, %zu, %zu)", i, j, k);
                }
    }
    {
        auto ptr = hv_sub1.sync().ptr<float>();
        ASSERT_EQ(TensorShape({S0 / 4, S1 / 2, S2 / 3}), hv_sub1.shape());
        for (size_t i = 0; i < S0 / 4; i ++)
            for (size_t j = 0; j < S1 / 2; j ++)
                for (size_t k = 0; k < S2 / 3; k ++) {
                    MGB_ASSERT_FLOAT_EQ(idx(i * 4, j * 2, k * 3), *(ptr ++)) <<
                        ssprintf("sub1: failed at (%zu, %zu, %zu)", i, j, k);
                }
    }
}

TEST(TestTensorStorage, CrossCNCopy2D) {
    auto cns = load_multiple_xpus(2);
    constexpr size_t S0 = 200, S1 = 500;
    HostTensorND hv{cns[0], dtype::Float32()};
    hv.resize({S0, S1});
    for (size_t i = 0; i < S0 * S1; ++ i)
        hv.ptr<float>()[i] = i;
    DeviceTensorND dev0;
    dev0.copy_from(hv).sync();

    bool failed = false;
    auto check = [&](size_t begin, size_t end) {
        ASSERT_FALSE(failed);
        failed = true;

        DeviceTensorND dev0_sub, dev1(cns[1]);
        dev0_sub = dev0.sub(Slice(begin, end).apply(dev0.layout(), 1));
        dev1.copy_from(dev0_sub);

        HostTensorND rst;
        rst.copy_from(dev1).sync();

        auto ptr = rst.ptr<float>();
        for (size_t i = 0; i < S0; ++ i)
            for (size_t j = begin; j < end; ++ j) {
                ASSERT_EQ(float(i * S1 + j), *ptr);
                ++ ptr;
            }

        failed = false;
    };

    check(0, 1);
    check(S1 - 1, S1);
    check(0, S1 - 1);
    check(1, S1);
    check(12, 21);
}

TEST(TestTensor, LayoutSlice) {
    TensorLayout ly0({4, 4, 4, 4}, dtype::Int32());

    auto ly = ly0;
    ly[1] = 2;
    auto sub = Slice(1, 3, 1).apply(ly0, 1);
    ASSERT_EQ(16u, sub.offset_elem());
    ASSERT_EQ(ly, sub.layout());

    ly0.init_contiguous_stride({1, 4, 4, 4});
    ly = ly0;
    ly[1] = 2;
    ly.stride[0] = 32;
    ly.stride[1] = 16;
    sub = Slice(1, 3, 1).apply(ly0, 1);
    ASSERT_EQ(16u, sub.offset_elem());
    ASSERT_EQ(ly, sub.layout());

    ly = ly0;
    ly[1] = 2;
    ly.stride[0] = -32;
    ly.stride[1] = -16;
    sub = Slice(3, 1, -1).apply(ly0, 1);
    ASSERT_EQ(48u, sub.offset_elem());
    ASSERT_EQ(ly, sub.layout());

    ly0.init_contiguous_stride({1, 4, 4, 4});
    ly = ly0;
    ly[1] = 1;
    ly.stride[0] = 16;
    ly.stride[1] = 16;
    sub = Slice(3, 4, 1).apply(ly0, 1);
    ASSERT_EQ(48u, sub.offset_elem());
    ASSERT_EQ(ly, sub.layout());
}

TEST(TestTensor, NoncontigCopyH2H) {
    run_noncontig_test<HostTensorND, HostTensorND>();
}

TEST(TestTensor, NoncontigCopyD2D) {
    run_noncontig_test<DeviceTensorND, DeviceTensorND>();
}

TEST(TestTensor, NoncontigCopyD2H) {
    run_noncontig_test<DeviceTensorND, HostTensorND>();
}

TEST(TestTensor, NoncontigCopyH2D) {
    run_noncontig_test<HostTensorND, DeviceTensorND>();
}

TEST(TestTensor, EmptyCheck) {
    HostTensorGenerator<> gen;
    auto hv = *gen({23});
    ASSERT_FALSE(hv.empty());
    hv.resize({});
    ASSERT_TRUE(hv.empty());
    hv.resize({2});
    ASSERT_FALSE(hv.empty());
    hv.resize({0});
    ASSERT_TRUE(hv.empty());
}

TEST(TestTensor, ValueDump) {
    HostTensorGenerator<> gen;
    auto val = debug::dump_tensor(*gen({23, 45}), "test");
    debug::write_to_file(output_file("TestTensor.ValueDump.bin").c_str(), val);
}

template <class Src, class Dst>
void run_negative_index_test() {
    constexpr size_t S0 = 200, S1 = 200;
    HostTensorND hv_init{CompNode::load("xpu0"), dtype::Float32()};
    hv_init.resize({S0, S1});
    for (size_t i = 0; i < S0 * S1; ++i)
        hv_init.ptr<float>()[i] = i;

    Src src;
    Src src_sub;
    Dst dst;
    auto check = [&](size_t begin, size_t end, int axis) {
        src.copy_from(hv_init).sync();
        src_sub = src.sub(Slice(begin, end).apply(src.layout(), axis));
        dst.copy_from(src_sub).sync();
        if (axis < 0)
            axis += 2;
        ASSERT_EQ(dst.layout().ndim, 2u);
        for (int i = 0; i < 2; i++) {
            if (i == axis)
                ASSERT_EQ(dst.layout()[i], end - begin);
            else
                ASSERT_EQ(dst.layout()[i], 200u);
        }
    };
    check(100, 200, -1);
    check(10, 20, -1);
    check(100, 200, -2);
    check(10, 20, -2);
    EXPECT_THROW(check(100, 200, -3), MegBrainError);
    EXPECT_THROW(check(10, 20, -3), MegBrainError);
    EXPECT_THROW(check(100, 200, 2), MegBrainError);
    EXPECT_THROW(check(10, 20, 2), MegBrainError);
}

TEST(TestTensor, NegativeIndex) {
    run_negative_index_test<HostTensorND, HostTensorND>();
    run_negative_index_test<DeviceTensorND, DeviceTensorND>();
    run_negative_index_test<DeviceTensorND, HostTensorND>();
    run_negative_index_test<HostTensorND, DeviceTensorND>();
}

TEST(TestTensor, CpuCudaD2DCopy) {
    REQUIRE_GPU(1);
    auto cn_cpu = CompNode::load("cpu0"),
         cn_gpu = CompNode::load("gpu0");

    HostTensorGenerator<> gen;
    constexpr size_t length = 233333;
    auto a = gen({length});
    for (auto config: {true, false}) {
        DeviceTensorND dev_a{cn_cpu}, dev_b{cn_gpu, a->shape(), a->dtype()};
        dev_a.copy_from(*a).sync();

        if (!config) {
            auto subspec = Slice(0, length, 3).apply(a->layout(), 0);
            dev_a = dev_a.sub(subspec);
            dev_b = dev_b.sub(subspec);
        }

        auto iadd = [ptr = dev_a.ptr<float>(), length = dev_a.shape()[0],
                stride = dev_a.layout().stride[0]]() {
            for (size_t i = 0; i < length; ++ i) {
                ptr[i * stride] += 1;
            }
        };
        CompNodeEnv::from_comp_node(cn_cpu).cpu_env().dispatch(iadd);
        auto event = cn_cpu.create_event();
        event->record();
        cn_gpu.device_wait_event(*event);
        dev_b.copy_from_fixlayout(dev_a);
        HostTensorND res;
        res.copy_from(dev_b).sync();
        MGB_ASSERT_TENSOR_EQ(HostTensorND::make_proxy(dev_a), res);
    }
}

TEST(TestTensor, ProxyToDefaultCPU) {
    auto cn = CompNode::load("xpux");
    auto x = HostTensorND(cn, TensorLayout({1, 2, 3}, dtype::Float32{}));
    auto y = x.proxy_to_default_cpu();
    ASSERT_EQ(y.comp_node(), CompNode::default_cpu());
    ASSERT_EQ(x.layout(), y.layout());
    ASSERT_EQ(x.raw_ptr(), y.raw_ptr());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
