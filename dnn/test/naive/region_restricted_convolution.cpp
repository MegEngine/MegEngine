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
template <typename rtype>
void mask_tensor_kernel(
        const TensorND& in, TensorND& out, const TensorND& mask,
        const int32_t mask_val) {
    megdnn_assert(
            in.layout.ndim == out.layout.ndim && in.layout.ndim == 4 &&
            mask.layout.ndim == 3);
    megdnn_assert_eq_layout(in.layout, out.layout);
    megdnn_assert(
            mask.layout[0] == in.layout[0] && mask.layout[1] == in.layout[2] &&
            mask.layout[2] == in.layout[3]);

    rtype* mask_ptr = mask.compatible_ptr<rtype>();
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

void mask_tensor(
        const TensorND& in, TensorND& out, const TensorND& mask,
        const int32_t mask_val) {
    if (mask.layout.dtype == dtype::Int32()) {
        mask_tensor_kernel<dt_int32>(in, out, mask, mask_val);
    } else if (mask.layout.dtype == dtype::Uint8()) {
        mask_tensor_kernel<dt_uint8>(in, out, mask, mask_val);
    }
}
}  // namespace

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_FORWARD) {
    Checker<RegionRestrictedConvolution> checker(handle());
    RegionRestrictedConvolution::Param param;
    constexpr int N = 3;

    UniformIntRNG rng{0, N - 1};

    auto extra_impl = [&, this](const TensorNDArray& tensors) {
        auto conv = handle()->create_operator<Convolution>();
        conv->param() = param;
        auto workspace_size = conv->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[4].layout, nullptr);
        dt_byte* workspace_ptr = static_cast<dt_byte*>(malloc(workspace_size));
        Workspace workspace{workspace_ptr, workspace_size};

        TensorND masked_src(
                malloc(tensors[0].layout.span().dist_byte()), tensors[0].layout);
        TensorNDArray dst_tensors;
        for (int i = 0; i < N; ++i) {
            dst_tensors.emplace_back(
                    malloc(tensors[4].layout.span().dist_byte()), tensors[4].layout);
        }
        for (int i = 0; i < N; ++i) {
            mask_tensor(tensors[0], masked_src, tensors[2], i);
            conv->exec(masked_src, tensors[1], dst_tensors[i], nullptr, workspace);
            mask_tensor(dst_tensors[i], dst_tensors[i], tensors[3], i);
        }
        free(workspace_ptr);

        using Mode = ElemwiseForward::Param::Mode;
        auto add = handle()->create_operator<ElemwiseForward>();
        add->param().mode = Mode::ADD;
        add->exec({dst_tensors[0], dst_tensors[1]}, tensors[4]);
        for (int i = 2; i < N; ++i) {
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

    checker.set_dtype(2, dtype::Uint8()).set_dtype(3, dtype::Uint8());

    checker.execs({{1, 8, 2, 2}, {4, 8, 1, 1}, {1, 2, 2}, {1, 2, 2}, {}})
            .execs({{20, 12, 30, 30}, {4, 12, 1, 1}, {20, 30, 30}, {20, 30, 30}, {}})
            .execs({{20, 8, 30, 30}, {4, 8, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}});

    param.sparse = Convolution::Param::Sparse::GROUP;
    checker.set_param(param)
            .execs({{20, 15, 30, 30}, {5, 4, 3, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}})
            .execs({{20, 25, 30, 30},
                    {25, 1, 1, 3, 3},
                    {20, 30, 30},
                    {20, 28, 28},
                    {}});

    checker.set_dtype(2, dtype::Int32()).set_dtype(3, dtype::Int32());
    checker.execs({{20, 15, 30, 30}, {5, 4, 3, 3, 3}, {20, 30, 30}, {20, 28, 28}, {}})
            .execs({{20, 25, 30, 30},
                    {25, 1, 1, 3, 3},
                    {20, 30, 30},
                    {20, 28, 28},
                    {}});
}

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_FORWARD_DENSE_BRUTE) {
    Checker<RegionRestrictedConvolutionForward> checker(handle());
    RegionRestrictedConvolutionForward::Param param;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(  // src
                            {1, 1, 4, 4}, dtype::Float32(),
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
                    TensorValue(  // filter
                            {1, 1, 2, 2}, dtype::Float32(), {1, 1, 1, 1}),
                    TensorValue(  // rin
                            {1, 4, 4}, dtype::Int32(),
                            {1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1}),
                    TensorValue(  // rout
                            {1, 3, 3}, dtype::Int32(), {0, 1, 1, 1, 0, 0, 1, 0, 1}),
                    {},  // output
            },
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {1, 1, 3, 3}, dtype::Float32(),
                            {4, 14, 18, 5, 9, 0, 13, 9, 50})});
}

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_BWD_DATA_DENSE_BRUTE) {
    Checker<RegionRestrictedConvolutionBackwardData> checker(handle());
    RegionRestrictedConvolutionBackwardData::Param param;
    checker.set_param(param).exect(
            Testcase{
                    // filter
                    TensorValue(
                            {1, 1, 2, 2},      // shape
                            dtype::Float32(),  // dtype
                            {1.f, 1.f, 1.f, 1.f}),
                    // diff
                    TensorValue(
                            {1, 1, 3, 3}, dtype::Float32(),
                            {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
                    // rin
                    TensorValue(
                            {1, 4, 4}, dtype::Int32(),
                            {1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1}),
                    // rout
                    TensorValue({1, 3, 3}, dtype::Int32(), {0, 1, 1, 1, 0, 0, 1, 0, 1}),
                    // grad
                    {}},
            Testcase{// filter
                     {},
                     // diff
                     {},
                     // rin
                     {},
                     // rout
                     {},
                     // grad
                     TensorValue(
                             {1, 1, 4, 4}, dtype::Float32(),
                             {0., 2., 5., 3., 1., 6., 5., 3., 0., 13., 9., 9., 0., 7.,
                              9., 9.})});
}

TEST_F(NAIVE, REGIONRESTRICTEDCONVOLUTION_BWD_DATA_GROUP_BRUTE) {
    Checker<RegionRestrictedConvolutionBackwardData> checker(handle());

    // params
    RegionRestrictedConvolutionBackwardData::Param param;
    param.sparse = RegionRestrictedConvolutionBackwardData::Param::Sparse::GROUP;
    param.mode = RegionRestrictedConvolutionBackwardData::Mode::CROSS_CORRELATION;
    param.compute_mode =
            RegionRestrictedConvolutionBackwardData::Param::ComputeMode::DEFAULT;
    param.pad_h = param.pad_w =
            0;  // forward param, naive backward data doesn't matter with deconv padding
    param.stride_h = param.stride_w = 1;

    // checker setting
    checker.set_param(param).exect(
            Testcase{// filter
                     TensorValue(
                             {2, 1, 1, 2, 2},   // shape
                             dtype::Float32(),  // dtype
                             {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f}),
                     // diff
                     TensorValue({1, 2, 1, 1}, dtype::Float32(), {1, 2}),
                     // rin
                     TensorValue({1, 2, 2}, dtype::Int32(), {1, 1, 1, 1}),
                     // rout
                     TensorValue({1, 1, 1}, dtype::Int32(), {1}),
                     // grad
                     {}},
            Testcase{// filter
                     {},
                     // diff
                     {},
                     // rin
                     {},
                     // rout
                     {},
                     // grad
                     TensorValue(
                             {1, 2, 2, 2}, dtype::Float32(),
                             {1, 2, 3, 4, 10, 12, 14, 16})});
}

// vim: syntax=cpp.doxygen
