#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/conv_bias.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#include <cudnn.h>

#define V1(x) #x
#define V(x)  V1(x)
#define CUDNN_VERSION_STRING \
    "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)

namespace megdnn {
namespace test {

TEST_F(CUDA, REGION_RESTRICTED_CONV_FORWARD_LARGE_FILTER) {
    Checker<RegionRestrictedConvolutionForward> checker(handle_cuda());
    auto opr = handle_cuda()->create_operator<ConvolutionForward>();
    for (auto dt : std::vector<DType>{dtype::Int32(), dtype::Uint8()}) {
        auto run = [&checker, &dt, &opr](
                           size_t n, size_t g, size_t h, size_t fh, size_t padding,
                           size_t stride) {
            RegionRestrictedConvolution::Param cur_param;
            cur_param.mode =
                    RegionRestrictedConvolution::Param::Mode::CROSS_CORRELATION;
            cur_param.sparse = RegionRestrictedConvolution::Param::Sparse::GROUP;
            checker.set_dtype(2, dt).set_dtype(3, dt);
            float scale = 64.f / sqrt(fh * fh);
            UniformFloatRNG rng(scale, 2 * scale);
            UniformIntRNG r_rng{0, 2};
            checker.set_rng(0, &rng).set_rng(1, &rng).set_rng(2, &r_rng).set_rng(
                    3, &r_rng);
            if (dt.enumv() == DTypeEnum::Float16) {
                checker.set_epsilon(1e-1);
            }

            cur_param.pad_h = cur_param.pad_w = padding;
            cur_param.stride_h = cur_param.stride_w = stride;

            size_t ho = infer_conv_shape(h, fh, stride, padding);

            checker.set_param(cur_param).execs(
                    {{n, g, h, h}, {g, 1, 1, fh, fh}, {n, h, h}, {n, ho, ho}, {}});
        };
        run(4, 8, 32, 3, 3 / 2, 1);
        run(4, 8, 32, 5, 5 / 2, 1);
        run(4, 8, 32, 7, 7 / 2, 1);
        run(1, 2, 32, 9, 9 / 2, 1);
        run(4, 8, 32, 11, 11 / 2, 1);
        run(4, 8, 32, 13, 13 / 2, 1);
        run(4, 8, 32, 15, 15 / 2, 1);
        run(4, 8, 32, 17, 17 / 2, 1);
        run(4, 8, 32, 19, 19 / 2, 1);
        run(4, 8, 32, 21, 21 / 2, 1);
        run(4, 8, 32, 23, 23 / 2, 1);
        run(4, 8, 32, 25, 25 / 2, 1);
        run(4, 8, 32, 27, 27 / 2, 1);
        run(4, 8, 32, 29, 29 / 2, 1);
        run(4, 8, 32, 31, 31 / 2, 1);
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, BENCHMARK_REGION_RESTRICTED_CONV_FORWARD_LARGE_FILTER_FP32) {
    require_compute_capability(7, 5);
    Benchmarker<ConvBiasForward> bencher(handle_cuda());
    bencher.set_display(false);
    bencher.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                    "DEPTHWISE_LARGE_FILTER", {})
                    .c_str()));

    Benchmarker<RegionRestrictedConvolutionForward> rr_bencher(handle_cuda());
    rr_bencher.set_display(false);

    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    using NonlineMode = ConvBias::Param::NonlineMode;
    param.nonlineMode = NonlineMode::IDENTITY;
    param.sparse = ConvBias::Param::Sparse::GROUP;

    RegionRestrictedConvolutionForward::Param rr_param;
    rr_param.format = RegionRestrictedConvolutionForward::Param::Format::NCHW;
    rr_param.sparse = RegionRestrictedConvolutionForward::Param::Sparse::GROUP;

    UniformIntRNG r_rng{0, 2};

    auto run_bench = [&](size_t batch, size_t g, size_t hi, size_t wi, size_t fh,
                         size_t fw, size_t sh, size_t sw, size_t nr_times) {
        param.pad_h = fh / 2;
        param.pad_w = fw / 2;
        param.stride_h = sh;
        param.stride_w = sw;

        rr_param.pad_h = fh / 2;
        rr_param.pad_w = fw / 2;
        rr_param.stride_h = sh;
        rr_param.stride_w = sw;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(4, dtype::Float32());
        bencher.set_times(nr_times);

        rr_bencher.set_param(rr_param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Int32())
                .set_dtype(3, dtype::Int32());
        rr_bencher.set_rng(2, &r_rng).set_rng(3, &r_rng).set_rng(0, &r_rng);
        rr_bencher.set_times(nr_times);

        size_t ho = infer_conv_shape(hi, fh, sh, param.pad_h);
        size_t wo = infer_conv_shape(wi, fw, sw, param.pad_w);
        TensorShape inp{batch, g, hi, wi}, kern{g, 1, 1, fh, fw}, rin{batch, hi, wi},
                rout{batch, ho, wo}, out{batch, g, ho, wo};

        float bandwith = static_cast<float>(
                                 inp.total_nr_elems() + kern.total_nr_elems() +
                                 out.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        float rr_bandwith = static_cast<float>(
                                    inp.total_nr_elems() + kern.total_nr_elems() +
                                    rin.total_nr_elems() + rout.total_nr_elems() +
                                    out.total_nr_elems()) /
                            (1024 * 1024 * 1024) * 1e3;

        auto time_in_ms = bencher.execs({inp, kern, {}, {}, out}) / nr_times;
        auto ops = 2.0 * batch * g * ho * wo * fh * fw / (time_in_ms * 1e-3) * 1e-12;

        auto rr_time_in_ms = rr_bencher.execs({inp, kern, rin, rout, out}) / nr_times;
        auto rr_ops =
                2.0 * batch * g * ho * wo * fh * fw / (rr_time_in_ms * 1e-3) * 1e-12;
        printf("RegionRestrictedDepthwiseLargeFilter vs DepthwiseLargeFilter: inp=%s, "
               "kern=%s, out=%s\n"
               "time: %.2f ms, time(rr): %.2f ms, perf: %.2fTops, perf(rr): %.2f Tops\n"
               "bandwidth: %.2fGB/s, bandwidth(rr): %.2fGB/s, speedup: %.2f.\n",
               inp.to_string().c_str(), kern.to_string().c_str(),
               out.to_string().c_str(), time_in_ms, rr_time_in_ms, ops, rr_ops,
               bandwith * 4 / time_in_ms, rr_bandwith * 4 / rr_time_in_ms,
               time_in_ms / rr_time_in_ms);
    };

    run_bench(64, 384, 32, 32, 3, 3, 1, 1, 10);
    run_bench(64, 384, 32, 32, 5, 5, 1, 1, 10);
    run_bench(64, 384, 32, 32, 7, 7, 1, 1, 10);
    run_bench(64, 384, 32, 32, 9, 9, 1, 1, 10);
    run_bench(64, 384, 32, 32, 11, 11, 1, 1, 10);
    run_bench(64, 384, 32, 32, 13, 13, 1, 1, 10);
    run_bench(64, 384, 32, 32, 15, 15, 1, 1, 10);
    run_bench(64, 384, 32, 32, 17, 17, 1, 1, 10);
    run_bench(64, 384, 32, 32, 19, 19, 1, 1, 10);
    run_bench(64, 384, 32, 32, 21, 21, 1, 1, 10);
    run_bench(64, 384, 32, 32, 23, 23, 1, 1, 10);
    run_bench(64, 384, 32, 32, 25, 25, 1, 1, 10);
    run_bench(64, 384, 32, 32, 27, 27, 1, 1, 10);
    run_bench(64, 384, 32, 32, 29, 29, 1, 1, 10);
    run_bench(64, 384, 32, 32, 31, 31, 1, 1, 10);
}

TEST_F(CUDA, BENCHMARK_REGION_RESTRICTED_CONV_FORWARD_LARGE_FILTER_UINT8) {
    require_compute_capability(7, 5);
    Benchmarker<ConvBiasForward> bencher(handle_cuda());
    bencher.set_display(false);
    bencher.set_before_exec_callback(conv_bias::ConvBiasAlgoChecker<ConvBiasForward>(
            ConvBiasForward::algo_name<ConvBiasForward::DirectParam>(
                    "DEPTHWISE_LARGE_FILTER", {})
                    .c_str()));

    Benchmarker<RegionRestrictedConvolutionForward> rr_bencher(handle_cuda());
    rr_bencher.set_display(false);

    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;
    using NonlineMode = ConvBias::Param::NonlineMode;
    param.nonlineMode = NonlineMode::IDENTITY;
    param.sparse = ConvBias::Param::Sparse::GROUP;

    RegionRestrictedConvolutionForward::Param rr_param;
    rr_param.format = RegionRestrictedConvolutionForward::Param::Format::NCHW;
    rr_param.sparse = RegionRestrictedConvolutionForward::Param::Sparse::GROUP;

    UniformIntRNG r_rng{0, 2};

    auto run_bench = [&](size_t batch, size_t g, size_t hi, size_t wi, size_t fh,
                         size_t fw, size_t sh, size_t sw, size_t nr_times) {
        param.pad_h = fh / 2;
        param.pad_w = fw / 2;
        param.stride_h = sh;
        param.stride_w = sw;

        rr_param.pad_h = fh / 2;
        rr_param.pad_w = fw / 2;
        rr_param.stride_h = sh;
        rr_param.stride_w = sw;

        bencher.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_dtype(4, dtype::Float32());
        bencher.set_times(nr_times);

        rr_bencher.set_param(rr_param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Uint8())
                .set_dtype(3, dtype::Uint8());
        rr_bencher.set_rng(2, &r_rng).set_rng(3, &r_rng).set_rng(0, &r_rng);
        rr_bencher.set_times(nr_times);

        size_t ho = infer_conv_shape(hi, fh, sh, param.pad_h);
        size_t wo = infer_conv_shape(wi, fw, sw, param.pad_w);
        TensorShape inp{batch, g, hi, wi}, kern{g, 1, 1, fh, fw}, rin{batch, hi, wi},
                rout{batch, ho, wo}, out{batch, g, ho, wo};

        float bandwith = static_cast<float>(
                                 inp.total_nr_elems() + kern.total_nr_elems() +
                                 out.total_nr_elems()) /
                         (1024 * 1024 * 1024) * 1e3;

        float rr_bandwith = static_cast<float>(
                                    inp.total_nr_elems() + kern.total_nr_elems() +
                                    rin.total_nr_elems() + rout.total_nr_elems() +
                                    out.total_nr_elems()) /
                            (1024 * 1024 * 1024) * 1e3;

        auto time_in_ms = bencher.execs({inp, kern, {}, {}, out}) / nr_times;
        auto ops = 2.0 * batch * g * ho * wo * fh * fw / (time_in_ms * 1e-3) * 1e-12;

        auto rr_time_in_ms = rr_bencher.execs({inp, kern, rin, rout, out}) / nr_times;
        auto rr_ops =
                2.0 * batch * g * ho * wo * fh * fw / (rr_time_in_ms * 1e-3) * 1e-12;
        printf("RegionRestrictedDepthwiseLargeFilter vs DepthwiseLargeFilter: inp=%s, "
               "kern=%s, out=%s\n"
               "time: %.2f ms, time(rr): %.2f ms, perf: %.2fTops, perf(rr): %.2f Tops\n"
               "bandwidth: %.2fGB/s, bandwidth(rr): %.2fGB/s, speedup: %.2f.\n",
               inp.to_string().c_str(), kern.to_string().c_str(),
               out.to_string().c_str(), time_in_ms, rr_time_in_ms, ops, rr_ops,
               bandwith * 4 / time_in_ms, rr_bandwith * 4 / rr_time_in_ms,
               time_in_ms / rr_time_in_ms);
    };

    run_bench(64, 384, 32, 32, 3, 3, 1, 1, 10);
    run_bench(64, 384, 32, 32, 5, 5, 1, 1, 10);
    run_bench(64, 384, 32, 32, 7, 7, 1, 1, 10);
    run_bench(64, 384, 32, 32, 9, 9, 1, 1, 10);
    run_bench(64, 384, 32, 32, 11, 11, 1, 1, 10);
    run_bench(64, 384, 32, 32, 13, 13, 1, 1, 10);
    run_bench(64, 384, 32, 32, 15, 15, 1, 1, 10);
    run_bench(64, 384, 32, 32, 17, 17, 1, 1, 10);
    run_bench(64, 384, 32, 32, 19, 19, 1, 1, 10);
    run_bench(64, 384, 32, 32, 21, 21, 1, 1, 10);
    run_bench(64, 384, 32, 32, 23, 23, 1, 1, 10);
    run_bench(64, 384, 32, 32, 25, 25, 1, 1, 10);
    run_bench(64, 384, 32, 32, 27, 27, 1, 1, 10);
    run_bench(64, 384, 32, 32, 29, 29, 1, 1, 10);
    run_bench(64, 384, 32, 32, 31, 31, 1, 1, 10);
}

#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
