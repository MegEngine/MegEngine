#include "megdnn/dtype.h"
#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, DEFORMABLE_CONV_FWD) {
    Checker<DeformableConv> checker(handle());
    DeformableConv::Param param;

    UniformIntRNG im_rng{0, 4};
    UniformIntRNG filter_rng{0, 4};
    UniformIntRNG offset_rng{-2, 2};
    UniformIntRNG mask_rng{0, 1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng);

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});

    param.sparse = DeformableConv::Param::Sparse::DENSE;
    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 2, 3, 3},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {}});
    //! check algo interface
    auto opr = handle()->create_operator<DeformableConv>();
    auto i0 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i1 = megdnn::TensorLayout({2, 1, 1, 3, 3}, megdnn::dtype::Float32());
    auto i2 = megdnn::TensorLayout({1, 2 * 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i3 = megdnn::TensorLayout({1, 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto o = opr->get_algorithm_info_heuristic(i0, i1, i2, i3, {});
    auto kk = o.desc.name;
    printf("%s\n", kk.c_str());
}

TEST_F(NAIVE, DEFORMABLE_CONV_BWD_FILTER) {
    Checker<DeformableConvBackwardFilter> checker(handle());
    DeformableConv::Param param;

    UniformIntRNG im_rng{0, 4};
    UniformIntRNG offset_rng{-2, 2};
    UniformIntRNG mask_rng{0, 1};
    UniformIntRNG out_grad_rng{0, 1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &offset_rng)
            .set_rng(2, &mask_rng)
            .set_rng(3, &out_grad_rng);
    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {1, 2 * 2 * 3 * 3, 5, 5},
             {1, 2 * 3 * 3, 5, 5},
             {1, 2, 5, 5},
             {2, 1, 1, 3, 3}});

    //! check algo interface
    auto opr = handle()->create_operator<DeformableConvBackwardFilter>();
    auto i0 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i1 = megdnn::TensorLayout({1, 2 * 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i2 = megdnn::TensorLayout({1, 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i3 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i4 = megdnn::TensorLayout({2, 1, 1, 3, 3}, megdnn::dtype::Float32());

    auto o = opr->get_algorithm_info_heuristic(i0, i1, i2, i3, i4);
    auto kk = o.desc.name;
    printf("%s\n", kk.c_str());
}

TEST_F(NAIVE, DEFORMABLE_CONV_BWD_DATA) {
    Checker<DeformableConvBackwardData> checker(handle());
    DeformableConv::Param param;

    ConstValue im_rng{1};
    ConstValue filter_rng{0.99};
    ConstValue offset_rng{1.1};
    ConstValue mask_rng{1};
    ConstValue out_grad_rng{1};

    checker.set_rng(0, &im_rng)
            .set_rng(1, &filter_rng)
            .set_rng(2, &offset_rng)
            .set_rng(3, &mask_rng)
            .set_rng(4, &out_grad_rng);

    param.pad_h = 1;
    param.pad_w = 1;
    param.stride_h = 1;
    param.stride_w = 1;
    param.dilate_h = 1;
    param.dilate_w = 1;
    param.format = DeformableConv::Param::Format::NCHW;
    param.sparse = DeformableConv::Param::Sparse::GROUP;

    checker.set_param(param).execs(
            {{1, 2, 5, 5},
             {2, 1, 1, 3, 3},
             {1, 1 * 2 * 3 * 3, 5, 5},
             {1, 1 * 3 * 3, 5, 5},
             {1, 2, 5, 5},
             {1, 2, 5, 5},
             {1, 1 * 2 * 3 * 3, 5, 5},
             {1, 1 * 3 * 3, 5, 5}});
    //! check algo interface
    auto opr = handle()->create_operator<DeformableConvBackwardData>();
    auto i0 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i1 = megdnn::TensorLayout({2, 1, 1, 3, 3}, megdnn::dtype::Float32());
    auto i2 = megdnn::TensorLayout({1, 1 * 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i3 = megdnn::TensorLayout({1, 1 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i4 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i5 = megdnn::TensorLayout({1, 2, 5, 5}, megdnn::dtype::Float32());
    auto i6 = megdnn::TensorLayout({1, 1 * 2 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto i7 = megdnn::TensorLayout({1, 1 * 3 * 3, 5, 5}, megdnn::dtype::Float32());
    auto o = opr->get_algorithm_info_heuristic(i0, i1, i2, i3, i4, i5, i6, i7);
    auto kk = o.desc.name;
    printf("%s\n", kk.c_str());
}
// vim: syntax=cpp.doxygen
