#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"
// #include "test/common/regin_restricted_convolution.h"
#include "test/common/extra_impl_helper.h"
#include "test/common/random_state.h"

using namespace megdnn;
using namespace test;

namespace {

void mask_tensor(
        const TensorND& in, TensorND& out, const TensorND& mask,
        const int32_t mask_val) {
    megdnn_assert(
            in.layout.ndim == out.layout.ndim && in.layout.ndim == 4 &&
            mask.layout.ndim == 3);
    megdnn_assert_eq_layout(in.layout, out.layout);
    megdnn_assert(
            mask.layout[0] == in.layout[0] && mask.layout[1] == in.layout[2] &&
            mask.layout[2] == in.layout[3]);

    int32_t* mask_ptr = mask.ptr<int32_t>();
    float* src_ptr = in.compatible_ptr<float>();
    float* dst_ptr = out.compatible_ptr<float>();

    for (size_t n = 0; n < in.layout[0]; ++n) {
        for (size_t c = 0; c < in.layout[1]; ++c) {
            for (size_t h = 0; h < in.layout[2]; ++h) {
                for (size_t w = 0; w < in.layout[3]; ++w) {
                    size_t mask_off = n * mask.layout.stride[0] +
                                      h * mask.layout.stride[1] +
                                      w * mask.layout.stride[2];
                    size_t src_dst_off =
                            n * in.layout.stride[0] + c * in.layout.stride[1] +
                            h * in.layout.stride[2] + w * in.layout.stride[3];
                    if (mask_ptr[mask_off] == mask_val) {
                        dst_ptr[src_dst_off] = src_ptr[src_dst_off];
                    } else {
                        dst_ptr[src_dst_off] = 0.;
                    }
                }
            }
        }
    }
}
}  // namespace

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_FORWARD) {
    Checker<RegionRestrictedConvolution> checker(handle());
    RegionRestrictedConvolution::Param param;
    constexpr int N = 3;

    UniformIntRNG rng{0, N-1};

    auto extra_impl = [&, this](const TensorNDArray& tensors) {
        auto conv = handle()->create_operator<Convolution>();
        conv->param() = param;
        auto workspace_size = conv->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[4].layout, nullptr);
        dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
        Workspace workspace{workspace_ptr, workspace_size};

        TensorND masked_src(malloc(tensors[0].layout.span().dist_byte()), tensors[0].layout);
        TensorNDArray dst_tensors;
        for(int i=0; i<N; ++i) {
            dst_tensors.emplace_back(malloc(tensors[4].layout.span().dist_byte()), tensors[4].layout);
        }
        for(int i=0; i<N; ++i) {
            mask_tensor(tensors[0], masked_src, tensors[2], i);
            conv->exec(masked_src, tensors[1], dst_tensors[i], nullptr, workspace);
            mask_tensor(dst_tensors[i], dst_tensors[i], tensors[3], i);
            
        }
        free(workspace_ptr);
        
        using Mode = ElemwiseForward::Param::Mode;
        auto add = handle()->create_operator<ElemwiseForward>();
        add->param().mode = Mode::ADD;
        add->exec({dst_tensors[0], dst_tensors[1]}, tensors[4]);
        for (int i=2; i<N; ++i) {
            add->exec({dst_tensors[i], tensors[4]}, tensors[4]);
        }
    };

    checker.set_extra_opr_impl(extra_impl)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32());

    checker.execs({{1, 8, 2, 2}, {4, 8, 1, 1}, {1, 2, 2}, {1, 2, 2}, {}})
            .execs({{20, 12, 30, 30}, {4, 12, 1, 1}, {20, 30, 30}, {20, 30, 30}, {}})
            .execs({{20, 8, 30, 30}, {4, 8, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}});

    param.sparse = Convolution::Param::Sparse::GROUP;
    checker.set_param(param)
    .execs({{20, 15, 30, 30}, {5, 4, 3, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}})
    .execs({{20, 25, 30, 30}, {25, 1, 1, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}});
}

#if 0

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_BACKWARD_DATA) {
    Checker<RegionRestrictedConvolutionBackwardData> checker(handle());
    using Param = RegionRestrictedConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc, size_t fh,
                   size_t fw, size_t stride, size_t padding, size_t dilate = 1,
                   size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff = TensorLayout{{n, oc * group, oh, ow}, dtype::Float32()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Float32()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Float32()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param);
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode : {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

TEST_F(NAIVE, CONVOLUTION_BACKWARD_DATA) {
    Checker<RegionRestrictedConvolutionBackwardData> checker(handle());
    using Param = RegionRestrictedConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc, size_t fh,
                   size_t fw, size_t stride, size_t padding, size_t dilate = 1,
                   size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff = TensorLayout{{n, oc * group, oh, ow}, dtype::Float32()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Float32()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Float32()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param);
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode : {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}
#endif

// vim: syntax=cpp.doxygen
