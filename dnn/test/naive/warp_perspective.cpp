/**
 * \file dnn/test/naive/warp_perspective.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/cv.h"
#include "test/common/checker.h"
#include "test/common/warp_perspective.h"
#include "megdnn/tensor_format.h"
#include "test/common/benchmarker.h"
#include "test/common/extra_impl_helper.h"

using namespace megdnn;
using namespace test;

namespace {
class NanMatRNG : public RNG {
    void gen(const TensorND& tensor_) override {
        auto& gen = RandomState::generator();
        std::uniform_real_distribution<dt_float32> pdist3(1.9f, 2.1f);
        std::uniform_real_distribution<dt_float32> pdist(0.9f, 1.1f);
        std::uniform_real_distribution<dt_float32> pdisth(0.4f, 0.6f);
        std::uniform_real_distribution<dt_float32> ndist(-1.1f, -0.9f);
        std::uniform_real_distribution<dt_float32> ndist3(-2.1f, -1.9f);
        std::uniform_real_distribution<dt_float32> ndisth(-0.6f, -0.4f);
        std::uniform_int_distribution<int> dice(0, 5);
        float* ptr = tensor_.ptr<dt_float32>();
        auto N = tensor_.layout.shape[0];
        for (size_t n = 0; n < N; ++n) {
            for (size_t i = 0; i < 9; ++i) {
                switch (dice(gen)) {
                    case 0:
                        ptr[i] = pdist3(gen);
                        break;
                    case 1:
                        ptr[i] = pdist(gen);
                        break;
                    case 2:
                        ptr[i] = pdisth(gen);
                        break;
                    case 3:
                        ptr[i] = ndist(gen);
                        break;
                    case 4:
                        ptr[i] = ndist3(gen);
                        break;
                    case 5:
                        ptr[i] = ndisth(gen);
                        break;
                }
            }
            ptr[6] = 1;
            ptr[7] = -1;
            ptr[8] = 5;
            ptr += 9;
        }
    }
};
}  // namespace

TEST_F(NAIVE, WARP_PERSPECTIVE_NCHW4) {
    using Param = WarpPerspective::Param;

    auto convert_true_format = [](const TensorLayout& layout) {
        if (layout.ndim == 4)
            return layout
                    .reshape(
                            {layout[0], layout[1] / 4, layout[2], layout[3], 4})
                    .dimshuffle({0, 1, 4, 2, 3});
        else
            return layout;
    };

    WarpPerspective::Param param;
    auto extra_impl = [&param, this,
                       convert_true_format](const TensorNDArray& tensors) {
        auto warp_perspective = handle()->create_operator<WarpPerspective>();
        warp_perspective->param() = param;
        warp_perspective->param().format = Param::Format::NCHW;

        TensorNDArray nchw_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = tensors[i].layout;
            if (layout.dtype.enumv() == DTypeEnum::QuantizedS8)
                layout.dtype = dtype::Int8();
            if (layout.ndim == 5) {
                layout = layout.reshape({layout[0], layout[1] * layout[4],
                                         layout[2], layout[3]});
            }
            nchw_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                      layout);
        }
        TensorNDArray nchw4_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = convert_true_format(nchw_tensors[i].layout);
            nchw4_tensors.emplace_back(tensors[i].raw_ptr, std::move(layout));
        }

        auto workspace_size = warp_perspective->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout);
        dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
        Workspace workspace{workspace_ptr, workspace_size};

        auto relayout = handle()->create_operator<RelayoutForward>();
        relayout->exec(nchw4_tensors[0], nchw_tensors[0]);
        relayout->exec(nchw4_tensors[1], nchw_tensors[1]);

        warp_perspective->exec(nchw_tensors[0], nchw_tensors[1],
                               nchw_tensors[2], workspace);

        relayout->exec(nchw_tensors[2], nchw4_tensors[2]);

        free(workspace_ptr);
        for (auto&& tensor : nchw_tensors) {
            free(tensor.raw_ptr);
        }
    };

    Checker<WarpPerspectiveForward> checker(handle());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    checker.set_extra_opr_impl(extra_impl);
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NCHW4;
        checker.set_param(param);
        checker.execs({{2, 1, 10, 11, 4}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{20, 300, 10, 11, 4}, {20, 3, 3}, {20, 300, 11, 12, 4}});
        checker.execs(
                {{2200, 3, 10, 11, 4}, {2200, 3, 3}, {2200, 3, 11, 12, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 25, 510, 4}});
        checker.execs({{1, 25, 25, 510, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 51, 51, 4}});
        checker.execs({{1, 25, 51, 51, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
        break;
    }
}

TEST_F(NAIVE, WARP_PERSPECTIVE) {
    Checker<WarpPerspective> checker(handle(), false);
    WarpPerspective::Param param;
    param.bmode = WarpPerspective::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpPerspective::Param::InterpolationMode::LINEAR;
    param.format = WarpPerspective::Param::Format::NCHW;

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3}, dtype::Uint8{},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 3, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f, 1.3f,
                                  1.5f, 3.0f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2}, dtype::Uint8{},
                                 {156, 183, 181, 195})});

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 3, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f, 1.3f,
                                  1.5f, 3.0f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {156, 183, 181, 195})});
}

TEST_F(NAIVE_MULTI_THREADS, WARP_PERSPECTIVE_NCHW4) {
    using Param = WarpPerspective::Param;

    auto convert_true_format = [](const TensorLayout& layout) {
        if (layout.ndim == 4)
            return layout
                    .reshape(
                            {layout[0], layout[1] / 4, layout[2], layout[3], 4})
                    .dimshuffle({0, 1, 4, 2, 3});
        else
            return layout;
    };

    WarpPerspective::Param param;
    auto extra_impl = [&param, this,
                       convert_true_format](const TensorNDArray& tensors) {
        auto warp_perspective = handle()->create_operator<WarpPerspective>();
        warp_perspective->param() = param;
        warp_perspective->param().format = Param::Format::NCHW;

        TensorNDArray nchw_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = tensors[i].layout;
            if (layout.dtype.enumv() == DTypeEnum::QuantizedS8)
                layout.dtype = dtype::Int8();
            if (layout.ndim == 5) {
                layout = layout.reshape({layout[0], layout[1] * layout[4],
                                         layout[2], layout[3]});
            }
            nchw_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                      layout);
        }
        TensorNDArray nchw4_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = convert_true_format(nchw_tensors[i].layout);
            nchw4_tensors.emplace_back(tensors[i].raw_ptr, std::move(layout));
        }

        auto workspace_size = warp_perspective->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout);
        dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
        Workspace workspace{workspace_ptr, workspace_size};

        auto relayout = handle()->create_operator<RelayoutForward>();
        relayout->exec(nchw4_tensors[0], nchw_tensors[0]);
        relayout->exec(nchw4_tensors[1], nchw_tensors[1]);

        warp_perspective->exec(nchw_tensors[0], nchw_tensors[1],
                               nchw_tensors[2], workspace);

        relayout->exec(nchw_tensors[2], nchw4_tensors[2]);

        free(workspace_ptr);
        for (auto&& tensor : nchw_tensors) {
            free(tensor.raw_ptr);
        }
    };

    Checker<WarpPerspectiveForward> checker(handle());
    WarpPerspectiveMatRNG rng;
    checker.set_rng(1, &rng);
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    checker.set_dtype(2, dtype::QuantizedS8(0.1f));
    checker.set_extra_opr_impl(extra_impl);
    for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                       WarpPerspective::BorderMode::REFLECT,
                       WarpPerspective::BorderMode::REPLICATE,
                       WarpPerspective::BorderMode::CONSTANT}) {
        param.border_val = 0.3f;
        param.bmode = bmode;
        param.imode = Param::InterpolationMode::LINEAR;

        param.format = Param::Format::NCHW4;
        checker.set_param(param);
        checker.execs({{2, 1, 10, 11, 4}, {2, 3, 3}, {2, 1, 11, 12, 4}});
        checker.execs({{20, 300, 10, 11, 4}, {20, 3, 3}, {20, 300, 11, 12, 4}});
        checker.execs(
                {{2200, 3, 10, 11, 4}, {2200, 3, 3}, {2200, 3, 11, 12, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 25, 510, 4}});
        checker.execs({{1, 25, 25, 510, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
        checker.execs({{1, 25, 25, 25, 4}, {1, 3, 3}, {1, 25, 51, 51, 4}});
        checker.execs({{1, 25, 51, 51, 4}, {1, 3, 3}, {1, 25, 25, 25, 4}});
        break;
    }
}

TEST_F(NAIVE_MULTI_THREADS, WARP_PERSPECTIVE) {
    Checker<WarpPerspective> checker(handle(), false);
    WarpPerspective::Param param;
    param.bmode = WarpPerspective::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpPerspective::Param::InterpolationMode::LINEAR;
    param.format = WarpPerspective::Param::Format::NCHW;

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3}, dtype::Uint8{},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 3, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f, 1.3f,
                                  1.5f, 3.0f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2}, dtype::Uint8{},
                                 {156, 183, 181, 195})});

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 3, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f, 1.3f,
                                  1.5f, 3.0f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {156, 183, 181, 195})});
}

TEST_F(NAIVE_MULTI_THREADS, WARP_PERSPECTIVE_FORWARD_HWCD4) {
    auto handle_multi_thread = handle();
    Checker<WarpPerspective> checker(handle(), false);
    TensorFormat img_fmt =
            Image2DPack4TensorFormat::make(2, handle_multi_thread);
    checker.set_fmt(0, img_fmt).set_fmt(2, img_fmt);
    for (auto dtype : std::vector<DType>{
                 dtype::Float32(), dtype::Float16(), dtype::QuantizedS8(4.3f),
                 dtype::Quantized8Asymm(2.4f, static_cast<uint8_t>(10))}) {
        for (auto bmode : {WarpPerspective::BorderMode::WRAP,
                           WarpPerspective::BorderMode::REFLECT,
                           WarpPerspective::BorderMode::CONSTANT,
                           WarpPerspective::BorderMode::REPLICATE,
                           WarpPerspective::BorderMode::CONSTANT}) {
            WarpPerspectiveMatRNG rng;
            checker.set_rng(1, &rng);
            WarpPerspective::Param param;
            param.border_val = 0.3f;
            param.bmode = bmode;
            param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
            param.format = param::WarpPerspective::Format::NHWCD4;
            if (dtype == dtype::Float16()) {
                //! if exists error, the value of a result pixel maybe another
                //! pixel in the origin image, so we just consider the avg error
                checker.set_epsilon(2e-1);
                checker.set_max_avg_error(1e-2);
            }
            checker.set_param(param);
            checker.set_dtype(0, dtype);
            checker.set_dtype(2, dtype);
            if (dtype.category() == DTypeCategory::FLOAT) {
                checker.set_dtype(1, dtype);
            } else {
                checker.set_dtype(1, dtype::Float32());
            }
            checker.execs({{2, 10, 1, 11, 4}, {2, 3, 3}, {2, 11, 1, 12, 4}});
            checker.execs({{22, 10, 1, 11, 4}, {22, 3, 3}, {22, 11, 1, 12, 4}});
        }
    }
#if MEGDNN_TEST_ASAN
//! asan detect nan will make test failed
#else
    // nan case
    NanMatRNG rng_nan;
    UniformFloatRNG rng_zero(0, 0);
    //! NanMatRng not support float16, I have to reset dtype to Float32
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32());
    for (auto rng : std::vector<RNG*>{&rng_nan, &rng_zero}) {
        param::WarpPerspective param;
        param.bmode = param::WarpPerspective::BorderMode::CONSTANT;
        param.imode = param::WarpPerspective::InterpolationMode::LINEAR;
        param.format = param::WarpPerspective::Format::NHWCD4;
        checker.set_rng(1, rng);
        param.border_val = 1.737;
        checker.set_param(param);
        checker.exec({{10, 10, 1, 11, 4}, {10, 3, 3}, {10, 12, 1, 13, 4}});
    }
#endif
}

#if MEGDNN_WITH_BENCHMARK
namespace {
void benchmark_impl(const typename WarpPerspective::Param& param,
                    std::vector<SmallVector<TensorShape>> shapes, size_t RUNS,
                    TaskExecutorConfig&& multi_thread_config,
                    TaskExecutorConfig&& single_thread_config) {
    std::vector<float> multi_thread_times, single_thread_times;
    {
        auto multi_thread_hanle =
                create_cpu_handle(0, true, &multi_thread_config);
        auto benchmarker =
                Benchmarker<WarpPerspective>(multi_thread_hanle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        for (auto shape : shapes) {
            multi_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    {
        auto single_thread_handle =
                create_cpu_handle(0, true, &single_thread_config);
        auto benchmarker =
                Benchmarker<WarpPerspective>(single_thread_handle.get());
        benchmarker.set_times(RUNS).set_display(false).set_param(param);
        for (auto shape : shapes) {
            single_thread_times.push_back(benchmarker.exec(shape) / RUNS);
        }
    }
    printf("Benchmark : Multi threads  %zu, ", multi_thread_config.nr_thread);
    printf("core_ids:");
    for (size_t i = 0; i < multi_thread_config.affinity_core_set.size(); i++) {
        printf("%zu ", multi_thread_config.affinity_core_set[i]);
    }
    printf(", Single thread core_id %zu\n",
           single_thread_config.affinity_core_set[0]);
    for (size_t i = 0; i < shapes.size(); i++) {
        auto shape = shapes[i];
        printf("Case: ");
        for (auto sh : shape)
            printf("%s ", sh.to_string().c_str());
        printf("%zu threads time: %f,\n single thread time: "
               "%f. spead up = %f, speedup/cores=%f\n",
               multi_thread_config.nr_thread, multi_thread_times[i],
               single_thread_times[i],
               single_thread_times[i] / multi_thread_times[i],
               single_thread_times[i] / multi_thread_times[i] /
                       multi_thread_config.nr_thread);
    }
}
}  // namespace

TEST_F(NAIVE_BENCHMARK_MULTI_THREADS, BENCHMARK_WARP_PERSPECTIVE) {
    constexpr size_t RUNS = 50;
    using BMode = param::WarpPerspective::BorderMode;
    using IMode = param::WarpPerspective::InterpolationMode;

    WarpPerspective::Param param;
    param.border_val = 0.3f;
    param.format = param::WarpPerspective::Format::NCHW;
    param.imode = IMode::INTER_LINEAR;
    param.bmode = BMode::REPLICATE;

    std::vector<SmallVector<TensorShape>> shapes;
    auto bench_case = [&](size_t N, size_t H, size_t W, size_t C) {
        SmallVector<TensorShape> shape{
                {N, C, H, W}, {N, 3, 3}, {N, C, 224, 224}};
        shapes.push_back(shape);
    };
    bench_case(1, 700, 490, 10);
    bench_case(1, 700, 490, 20);
    bench_case(1, 700, 490, 30);
    bench_case(1, 500, 334, 10);
    bench_case(1, 500, 334, 20);
    bench_case(1, 500, 334, 30);
    bench_case(1, 140, 144, 10);
    bench_case(1, 140, 144, 20);
    bench_case(1, 140, 114, 30);

    printf("Benchmark warp perspective\n");
    benchmark_impl(param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {4}});
    benchmark_impl(param, shapes, RUNS, {4, {4, 5, 6, 7}}, {1, {7}});
    benchmark_impl(param, shapes, RUNS, {2, {4, 5}}, {1, {4}});
}
#endif

TEST_F(NAIVE, WARP_PERSPECTIVE_BFLOAT16) {
    Checker<WarpPerspective> checker(handle(), false);
    WarpPerspective::Param p;
    p.bmode = WarpPerspective::Param::BorderMode::BORDER_REFLECT;
    p.imode = WarpPerspective::Param::InterpolationMode::LINEAR;
    p.format = WarpPerspective::Param::Format::NCHW;

    auto extra_impl = extra_impl_helper<WarpPerspective>(handle(), p);
    checker.set_param(p)
            .set_epsilon(1e-1)
            .set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::BFloat16())
            .set_extra_opr_impl(extra_impl)
            .execs({{1, 1, 3, 3}, {1, 3, 3}, {1, 1, 2, 2}})
            .execs({{1000, 2, 10, 11}, {1000, 3, 3}, {1000, 2, 12, 13}});
}

TEST_F(NAIVE, WARP_PERSPECTIVE_BACKWARD_DATA_BFLOAT16) {
    Checker<WarpPerspectiveBackwardData> checker(handle(), false);
    WarpPerspectiveBackwardData::Param p;
    p.bmode = WarpPerspectiveBackwardData::Param::BorderMode::BORDER_REFLECT;
    p.imode = WarpPerspectiveBackwardData::Param::InterpolationMode::LINEAR;
    p.format = WarpPerspectiveBackwardData::Param::Format::NCHW;

    auto extra_impl =
            extra_impl_helper<WarpPerspectiveBackwardData>(handle(), p);
    checker.set_param(p)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::BFloat16())
            .set_dtype(2, dtype::BFloat16())
            .set_extra_opr_impl(extra_impl)
            .set_epsilon(1e-1)
            .execs({{1, 3, 3}, {1, 1, 2, 2}, {1, 1, 3, 3}});
}

TEST_F(NAIVE, WARP_PERSPECTIVE_BACKWARD_MAT_BFLOAT16) {
    Checker<WarpPerspectiveBackwardMat> checker(handle(), false);
    WarpPerspectiveBackwardMat::Param p;
    p.bmode = WarpPerspectiveBackwardMat::Param::BorderMode::BORDER_REFLECT;
    p.imode = WarpPerspectiveBackwardMat::Param::InterpolationMode::LINEAR;
    p.format = WarpPerspectiveBackwardMat::Param::Format::NCHW;
    p.border_val = 0.3f;

    auto extra_impl =
            extra_impl_helper<WarpPerspectiveBackwardMat>(handle(), p);
    checker.set_param(p)
            .set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::BFloat16())
            .set_dtype(3, dtype::Float32())
            .set_extra_opr_impl(extra_impl)
            .set_epsilon(1e-1)
            .execs({{1000, 3, 11, 12},
                    {1000, 3, 3},
                    {1000, 3, 10, 11},
                    {1000, 3, 3}});
}

// vim: syntax=cpp.doxygen
