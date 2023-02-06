#include "megdnn/dtype.h"
#include "test/common/relayout.h"
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, GENERALNORM_FORWARD) {
    using Param = GeneralNormForward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-5;
    Checker<GeneralNormForward> checker(handle_cuda());
    checker.set_epsilon(1e-1);

    auto run_dtype = [&](DType d) {
        for (size_t A : {10, 30})
            for (size_t B : {10, 30}) {
                param.axis_start = 0;
                param.axis_end = 1;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{A, B, A, B}, {A}, {A}, {}, {}, {}});
                param.axis_start = 1;
                param.axis_end = 2;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{A, B, A, B}, {B}, {B}, {}, {}, {}});
                param.axis_start = 1;
                param.axis_end = 3;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{A, B, A, B}, {B, A}, {B, A}, {}, {}, {}});
                param.axis_start = 1;
                param.axis_end = 4;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{A, B, A, B}, {B, A, B}, {B, A, B}, {}, {}, {}});
            }
    };

    run_dtype(dtype::Float32());
    run_dtype(dtype::Float16());
    run_dtype(dtype::BFloat16());

    auto run_all_test = [&](TensorShape input, size_t start, size_t end) {
        param.axis_start = start;
        param.axis_end = end;
        TensorShape weight_shape;
        for (size_t i = 0; i < end - start; ++i) {
            weight_shape[i] = input[i + start];
        }
        weight_shape.ndim = end - start;
        param.affine = true;
        checker.set_param(param).execs({input, weight_shape, weight_shape, {}, {}, {}});
        param.affine = false;
        checker.set_param(param).execs({input, weight_shape, weight_shape, {}, {}, {}});
    };
    // run_all_test({3, 32, 128, 128}, 0, 1); //checker.set_epsilon(1e-1);
    run_all_test({3, 32, 128, 128}, 1, 2);
    run_all_test({3, 32, 128, 128}, 2, 3);
    run_all_test({3, 32, 128, 128}, 3, 4);
    run_all_test({3, 32, 128, 128}, 0, 2);
    run_all_test({3, 32, 128, 128}, 1, 3);
    run_all_test({3, 32, 128, 128}, 2, 4);
    run_all_test({3, 32, 128, 128}, 0, 3);
    run_all_test({3, 32, 128, 128}, 1, 4);
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_GENERALNORM_FORWARD_SPEED_FP32) {
    size_t iter = 10;

    auto benchmarker_layer = Benchmarker<LayerNormForward>(handle_cuda());
    LayerNormForward::Param param_layer;
    param_layer.affine = true;
    param_layer.eps = 1e-5;

    auto benchmarker_relayout = Benchmarker<Relayout>(handle_cuda());

    auto benchmarker_general = Benchmarker<GeneralNormForward>(handle_cuda());
    benchmarker_general.set_times(iter);
    GeneralNormForward::Param param_general;
    param_general.affine = true;
    param_general.eps = 1e-5;

    auto general_run = [&](TensorShape input, TensorShape weight, TensorShape bias,
                           size_t start, size_t end) {
        param_general.axis_start = start;
        param_general.axis_end = end;
        float totalTime = 0.f;
        totalTime = benchmarker_general.set_param(param_general)
                            .exec({input, weight, bias, {}, {}, {}}) /
                    iter;

        size_t num = 1;
        for (size_t i = 0; i < input.ndim; ++i)
            num *= input[i];
        float perf = 7.0f * num / totalTime * 1e-6;
        printf("[%zux%zux%zux%zu] ax[%ld~%ld] "
               "t: %.4fms (%.4f GFLOPs)\n",
               input[0], input[1], input[2], input[3], start, end - 1, totalTime, perf);
    };

    auto general_run_warp = [&](size_t N, size_t C, size_t H, size_t W) {
        general_run({N, C, H, W}, {N}, {N}, 0, 1);
        general_run({N, C, H, W}, {C}, {C}, 1, 2);
        general_run({N, C, H, W}, {H}, {H}, 2, 3);
        general_run({N, C, H, W}, {W}, {W}, 3, 4);
        general_run({N, C, H, W}, {W}, {W}, 3, 4);
        general_run({N, C, H, W}, {N, C}, {N, C}, 0, 2);
        general_run({N, C, H, W}, {C, H}, {C, H}, 1, 3);
        general_run({N, C, H, W}, {H, W}, {H, W}, 2, 4);
        general_run({N, C, H, W}, {H, W}, {H, W}, 2, 4);
        general_run({N, C, H, W}, {N, C, H}, {N, C, H}, 0, 3);
        general_run({N, C, H, W}, {C, H, W}, {C, H, W}, 1, 4);
        general_run({N, C, H, W}, {C, H, W}, {C, H, W}, 1, 4);
    };
    general_run_warp(3, 32, 100, 100);
    general_run_warp(3, 32, 1024, 1024);
    general_run_warp(10, 10, 10, 10);
    general_run_warp(20, 20, 20, 20);
    general_run_warp(40, 40, 40, 40);
    general_run_warp(80, 80, 80, 80);

    auto layer_vs_general_run = [&](TensorShape input, TensorShape relayout_input,
                                    TensorShape weight, size_t norm_dim,
                                    size_t norm_size, bool need_relayout, size_t start,
                                    size_t end) {
        param_layer.normalized_dim = norm_dim;
        param_layer.normalized_size = norm_size;
        benchmarker_layer.set_param(param_layer);
        param_general.axis_start = start;
        param_general.axis_end = end;
        benchmarker_general.set_param(param_general);

        float layer_totaltime = 0;
        float general_totaltime = 0;
        if (need_relayout)
            for (size_t i = 0; i < iter; ++i) {
                layer_totaltime += benchmarker_relayout.exec({input, relayout_input});
                layer_totaltime += benchmarker_layer.exec(
                        {relayout_input, weight, weight, {}, {}, {}});
                // layer_totaltime += benchmarker_relayout.exec({relayout_input, input});
            }
        else
            for (size_t i = 0; i < iter; ++i) {
                layer_totaltime += benchmarker_layer.exec(
                        {relayout_input, weight, weight, {}, {}, {}});
            }
        layer_totaltime /= iter;

        general_totaltime =
                benchmarker_general.exec({input, weight, weight, {}, {}, {}}) / iter;
        printf("[%zux%zux%zux%zu]", input[0], input[1], input[2], input[3]);
        printf(" [");
        for (auto i = start; i < end; i++)
            printf("%ld", i);
        printf("] ");
        printf("[generalnorm] %.4fms vs %.4fms [layernorm]%s\n", general_totaltime,
               layer_totaltime, need_relayout ? "[trans]" : "");
    };

    // layer_vs_general_run(input, relayout_input,
    //                      norm_shape_layer, norm_axis_layer, norm_size_layer,
    //                      need_relayout,
    //                      norm_axis_start_general, norm_axis_end_general,
    //                      )
    auto run = [&](size_t N, size_t C, size_t H, size_t W) {
        // 第一种情况：对最后几个轴进行 norm，此时的 layernorm 不需要进行
        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {C, H, W}, 3, C * H * W, false, 1, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {C, H, W}, 3, C * H * W, false, 1, 4);

        // 第二种情况：对中间几个轴进行 norm，此时的 layernorm 需要进行 transpose
        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run({N, C, H, W}, {N, C, W, H}, {H}, 1, H, true, 2, 3);
        layer_vs_general_run({N, C, H, W}, {N, H, W, C}, {C}, 1, C, true, 1, 2);
        layer_vs_general_run({N, C, H, W}, {C, H, W, N}, {N}, 1, N, true, 0, 1);

        layer_vs_general_run({N, C, H, W}, {N, C, H, W}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run({N, C, H, W}, {N, W, C, H}, {C, H}, 2, C * H, true, 1, 3);
        layer_vs_general_run({N, C, H, W}, {H, W, N, C}, {N, C}, 2, N * C, true, 0, 2);

        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {C, H, W}, 3, C * H * W, false, 1, 4);
        layer_vs_general_run(
                {N, C, H, W}, {W, N, C, H}, {N, C, H}, 3, N * C * H, true, 0, 3);
    };
    run(3, 3, 228, 228);
    run(3, 16, 228, 228);
    run(3, 32, 228, 228);
    run(3, 64, 228, 228);
    run(3, 128, 228, 228);
    run(3, 32, 100, 100);
    run(10, 10, 10, 10);
    run(30, 30, 30, 30);
    run(60, 60, 60, 60);
}
#endif

TEST_F(CUDA, GENERALNORM_BACKWARD) {
    using Param = GeneralNormBackward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-5;
    Checker<GeneralNormBackward> checker(handle_cuda());
    checker.set_epsilon(1e-1);

    auto run = [&](DType d) {
        for (size_t n_slices : {10, 30})
            for (size_t slice_len : {10, 30}) {
                param.axis_start = 0;
                param.axis_end = 1;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, dtype::Float32())
                        .set_dtype(4, dtype::Float32())
                        .execs({{n_slices, slice_len},
                                {n_slices, slice_len},
                                {n_slices},
                                {slice_len},
                                {slice_len},
                                {},
                                {},
                                {}});
                param.axis_start = 1;
                param.axis_end = 2;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, dtype::Float32())
                        .set_dtype(4, dtype::Float32())
                        .execs({{n_slices, slice_len},
                                {n_slices, slice_len},
                                {slice_len},
                                {n_slices},
                                {n_slices},
                                {},
                                {},
                                {}});
            }
    };
    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_GENERALNORM_BACKWARD_SPEED_FP32) {
    size_t iter = 10;

    auto benchmarker_layer = Benchmarker<LayerNormBackward>(handle_cuda());
    LayerNormBackward::Param param_layer;
    param_layer.affine = true;
    param_layer.eps = 1e-5;

    auto benchmarker_relayout = Benchmarker<Relayout>(handle_cuda());

    auto benchmarker_general = Benchmarker<GeneralNormBackward>(handle_cuda());
    benchmarker_general.set_times(iter);
    GeneralNormBackward::Param param_general;
    param_general.affine = true;
    param_general.eps = 1e-5;

    auto layer_vs_general_run = [&](TensorShape input, TensorShape relayout_input,
                                    TensorShape mean, TensorShape weight,
                                    size_t norm_dim, size_t norm_size,
                                    bool need_relayout, size_t start, size_t end) {
        param_layer.normalized_dim = norm_dim;
        param_layer.normalized_size = norm_size;
        benchmarker_layer.set_param(param_layer);
        param_general.axis_start = start;
        param_general.axis_end = end;
        benchmarker_general.set_param(param_general);

        float layer_totaltime = 0;
        float general_totaltime = 0;
        if (need_relayout)
            for (size_t i = 0; i < iter; ++i) {
                layer_totaltime += benchmarker_layer.exec(
                        {relayout_input,
                         relayout_input,
                         weight,
                         mean,
                         mean,
                         {},
                         {},
                         {}});
                layer_totaltime += benchmarker_relayout.exec({relayout_input, input});
            }
        else
            for (size_t i = 0; i < iter; ++i) {
                layer_totaltime += benchmarker_layer.exec(
                        {relayout_input,
                         relayout_input,
                         weight,
                         mean,
                         mean,
                         {},
                         {},
                         {}});
            }
        layer_totaltime /= iter;

        general_totaltime = benchmarker_general.exec(
                                    {input, input, weight, mean, mean, {}, {}, {}}) /
                            iter;
        printf("[%zux%zux%zux%zu]", input[0], input[1], input[2], input[3]);
        printf(" [");
        for (auto i = start; i < end; i++)
            printf("%ld", i);
        printf("] ");
        printf("[generalnorm] %.4fms vs %.4fms [layernorm]%s\n", general_totaltime,
               layer_totaltime, need_relayout ? "[trans]" : "");
    };

    // layer_vs_general_run(input, relayout_input, unnorm_shape_layer,
    //                      norm_shape_layer, norm_axis_layer, norm_size_layer,
    //                      need_relayout,
    //                      norm_axis_start_general, norm_axis_end_general)
    auto run = [&](size_t N, size_t C, size_t H, size_t W) {
        // 第一种情况：对最后几个轴进行 norm，此时的 layernorm 不需要进行
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C, H}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C, H}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N}, {C, H, W}, 3, C * H * W, false, 1, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N}, {C, H, W}, 3, C * H * W, false, 1, 4);

        // 第二种情况：对中间几个轴进行 norm，此时的 layernorm 需要进行 transpose
        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C, H}, {W}, 1, W, false, 3, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, C, W, H}, {N, C, W}, {H}, 1, H, true, 2, 3);
        layer_vs_general_run(
                {N, C, H, W}, {N, H, W, C}, {N, H, W}, {C}, 1, C, true, 1, 2);
        layer_vs_general_run(
                {N, C, H, W}, {C, H, W, N}, {C, H, W}, {N}, 1, N, true, 0, 1);

        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N, C}, {H, W}, 2, H * W, false, 2, 4);
        layer_vs_general_run(
                {N, C, H, W}, {N, W, C, H}, {N, W}, {C, H}, 2, C * H, true, 1, 3);
        layer_vs_general_run(
                {N, C, H, W}, {H, W, N, C}, {H, W}, {N, C}, 2, N * C, true, 0, 2);

        layer_vs_general_run(
                {N, C, H, W}, {N, C, H, W}, {N}, {C, H, W}, 3, C * H * W, false, 1, 4);
        layer_vs_general_run(
                {N, C, H, W}, {W, N, C, H}, {W}, {N, C, H}, 3, N * C * H, true, 0, 3);
    };
    run(3, 3, 228, 228);
    run(3, 16, 228, 228);
    run(3, 32, 228, 228);
    run(3, 64, 228, 228);
    run(3, 32, 100, 100);
    run(10, 10, 10, 10);
    run(30, 30, 30, 30);
    run(60, 60, 60, 60);
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
