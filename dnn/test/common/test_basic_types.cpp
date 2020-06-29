/**
 * \file dnn/test/common/test_basic_types.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "megdnn/tensor_format.h"

// clang-format off
#include "test/common/utils.h"
#include "test/common/fix_gtest_on_platforms_without_exception.inl"
// clang-format on

using namespace megdnn;
using megdnn::test::TensorReshapeError;
using megdnn::test::MegDNNError;

namespace {

TensorLayout make_layout(std::initializer_list<size_t> shape,
        std::initializer_list<ptrdiff_t> stride = {},
        DType dtype = {}) {
    TensorLayout rst;
    rst.init_contiguous_stride(TensorShape(shape));
    if (stride.size()) {
        megdnn_assert(stride.size() == shape.size());
        auto iter = stride.begin();
        for (size_t i = 0; i < rst.ndim; i ++)
            rst.stride[i] = *(iter ++);
        megdnn_assert(iter == stride.end());
    }
    rst.dtype = dtype;
    return rst;
}

} // anonymous namespace

#if MEGDNN_64_BIT
TEST(BASIC_TYPES, TOTAL_NR_ELEMS) {
    TensorShape shp{1u<<31, 1u<<31};
    ASSERT_EQ(size_t(1)<<62, shp.total_nr_elems());
    TensorLayout lt{dtype::Float32()};
    ASSERT_EQ(size_t(1)<<62, lt.init_contiguous_stride(shp));

    ASSERT_THROW(TensorShape(
                {size_t(1ull<<33), size_t(1ull<<32)}).total_nr_elems(),
            MegDNNError);
    ASSERT_THROW(TensorShape(
                {size_t(1ull<<32), size_t(1ull<<32)}).total_nr_elems(),
            MegDNNError);

    ASSERT_EQ(1ull<<63, TensorShape({size_t(1ull<<32), size_t(1ull<<31)}).total_nr_elems());
    ASSERT_EQ(static_cast<size_t>(105), TensorShape({5, 3, 7}).total_nr_elems());
    ASSERT_EQ(static_cast<size_t>(192), TensorShape({4, 6, 8}).total_nr_elems());
}
#endif

TEST(BASIC_TYPES, RESHAPE) {
    EXPECT_EQ(make_layout({2, 3, 4}).reshape({24}), make_layout({24}));
    EXPECT_EQ(make_layout({2, 3, 4}).reshape({2, 12}), make_layout({2, 12}));
    EXPECT_EQ(make_layout({2, 3, 4}).reshape({6, 4}), make_layout({6, 4}));
    EXPECT_EQ(make_layout({2, 3, 4}).reshape({1, 6, 1, 4, 1}),
            make_layout({1, 6, 1, 4, 1}));
#define EXPECT_THROWV(_v, _err) EXPECT_THROW([](auto x){return x;}(_v), _err)
// cast to void trick does not work for GCC 7

    EXPECT_THROWV(make_layout({2, 3}).reshape({5}), TensorReshapeError);

    EXPECT_EQ(make_layout({2, 3}, {6, 2}).reshape({6}), make_layout({6}, {2}));
    EXPECT_EQ(make_layout({3, 3, 3}, {2, 2, 2}).reshape({3, 3, 3}),
            make_layout({3, 3, 3}, {2, 2, 2}));
    EXPECT_THROWV(make_layout({2, 3}, {6, 1}).reshape({6}), TensorReshapeError);
    EXPECT_THROWV(make_layout({2, 3}, {0, 1}).reshape({6}), TensorReshapeError);
    EXPECT_THROWV(make_layout({2, 3}, {1, 0}).reshape({6}), TensorReshapeError);
    EXPECT_THROWV(make_layout({2, 3, 3, 2, 4, 3},
                {216, 72, 12, 36, 3, 1}).reshape(
                    {2, 3, 8, 9}), TensorReshapeError);

    EXPECT_EQ(make_layout({2, 3}, {0, 0}).reshape({6}), make_layout({6}, {0}));
    EXPECT_EQ(make_layout({2, 3, 2}, {0, 0, 1}).reshape({6, 2}),
            make_layout({6, 2}, {0, 1}));
}

TEST(BASIC_TYPES, COLLAPSE_CONTIGUOUS) {
    EXPECT_EQ(make_layout({3}, {2}),
            make_layout({3, 1}, {2, 5}).collapse_contiguous());
}

TEST(BASIC_TYPES, IS_CONTIGUOUS) {
    EXPECT_FALSE(make_layout({3, 2}, {3, 1}).is_contiguous());
    EXPECT_FALSE(make_layout({3, 2}, {3, 1}).is_contiguous_allow_brdcst());
    EXPECT_TRUE(make_layout({3, 2}, {2, 1}).is_contiguous());
    EXPECT_TRUE(make_layout({3, 2}, {2, 1}).is_contiguous_allow_brdcst());
    EXPECT_FALSE(make_layout({2, 3, 2}, {2, 0, 1}).is_contiguous());
    EXPECT_TRUE(make_layout({2, 3, 2}, {2, 0, 1}).is_contiguous_allow_brdcst());
}

TEST(BASIC_TYPES, IS_ABS_MONOTONOUS_ALLOW_BRDCST) {
    EXPECT_TRUE(make_layout({3, 2}, {3, 1}).is_abs_monotonous_allow_brdcst());
    EXPECT_TRUE(make_layout({3, 2}, {-3, -1}).is_abs_monotonous_allow_brdcst());
    EXPECT_TRUE(make_layout({3, 2}, {3, -1}).is_abs_monotonous_allow_brdcst());
    EXPECT_TRUE(
            make_layout({3, 2, 3}, {3, 0, 1}).is_abs_monotonous_allow_brdcst());
    EXPECT_TRUE(make_layout({3, 1, 3}, {3, 23, -1})
                        .is_abs_monotonous_allow_brdcst());
    EXPECT_FALSE(make_layout({3, 2}, {1, 3}).is_abs_monotonous_allow_brdcst());
    EXPECT_FALSE(
            make_layout({3, 2}, {-1, -3}).is_abs_monotonous_allow_brdcst());
    EXPECT_FALSE(
            make_layout({3, 2, 3}, {1, 0, 3}).is_abs_monotonous_allow_brdcst());
    EXPECT_FALSE(make_layout({3, 1, 3}, {1, 23, 3})
                        .is_abs_monotonous_allow_brdcst());
}

TEST(BASIC_TYPES, IS_NON_OVERLAP) {
    EXPECT_TRUE(make_layout({2, 3}, {3, 1}).is_non_overlapping_strong());
    EXPECT_TRUE(make_layout({3, 2}, {1, 3}).is_non_overlapping_strong());
    EXPECT_FALSE(make_layout({2, 3}, {4, 2}).is_non_overlapping_strong());
    EXPECT_TRUE(make_layout({2, 3}, {5, 2}).is_non_overlapping_strong());
    EXPECT_FALSE(make_layout({2, 3}, {3, 0}).is_non_overlapping_strong());
    EXPECT_TRUE(make_layout({2, 3}, {3, -1}).is_non_overlapping_strong());
    EXPECT_FALSE(make_layout({2, 3}, {2, 1}).is_non_overlapping_strong());
    EXPECT_TRUE(make_layout({1, 5}, {1, 1}).is_non_overlapping_strong());
}

TEST(BASIC_TYPES, EQ_SHAPE) {
    TensorShape shp0 = make_layout({2, 3});
    EXPECT_TRUE(shp0.eq_shape({2, 3}));
    EXPECT_FALSE(shp0.eq_shape({2, 4}));
    auto shp1 = shp0;
    -- shp1.ndim;
    EXPECT_FALSE(shp0.eq_shape(shp1));
}

TEST(BASIC_TYPES, EQ_LAYOUT) {
    auto ly0 = make_layout({2, 3});
    EXPECT_TRUE(ly0.eq_layout(make_layout({2, 3})));
    EXPECT_FALSE(ly0.eq_layout(make_layout({2, 3}, {2, 0})));
    EXPECT_FALSE(make_layout({2, 2}, {2, 2}).eq_layout(
                 make_layout({2, 2}, {2, 1})));
    // test for shape 1
    EXPECT_TRUE(make_layout({2, 1}, {2, 2}).eq_layout(
                make_layout({2, 1}, {2, 1})));
}

TEST(BASIC_TYPES, LAYOUT_SPAN) {
    auto span = make_layout({2, 3}, {-1, 1}, dtype::Float32{}).span();
    ASSERT_EQ(-1, span.low_elem);
    ASSERT_EQ(3, span.high_elem);
    ASSERT_EQ(-4, span.low_byte);
    ASSERT_EQ(12, span.high_byte);

    span = make_layout({2, 0}, {-1, 1}, dtype::Float32{}).span();
    ASSERT_EQ(0, span.low_elem);
    ASSERT_EQ(0, span.high_elem);
    ASSERT_EQ(0, span.low_byte);
    ASSERT_EQ(0, span.high_byte);
}

namespace {
    size_t image_width(const TensorLayout& layout) {
        return layout.format.as_impl<Image2DPack4TensorFormat>().image_width(layout);
    }
    size_t image_row_pitch(const TensorLayout& layout) {
        return layout.format.as_impl<Image2DPack4TensorFormat>().image_row_pitch(layout);
    }
    size_t image_height(const TensorLayout& layout) {
        return layout.format.as_impl<Image2DPack4TensorFormat>().image_height(layout);
    }
}

TEST(BASIC_TYPES, TENSOR_LAYOUT_FMT) {
    TensorFormat fmt = Image2DPack4TensorFormat::make_raw(1, 1024);
    ASSERT_FALSE(fmt.is_default());
    ASSERT_TRUE(TensorFormat{}.is_default());
    TensorLayout layout{{5, 3, 8}, dtype::Float32{}, fmt};

    ASSERT_EQ(layout.stride[2], 1);
    ASSERT_EQ(layout.stride[1], 8);
    ASSERT_EQ(layout.stride[0], 1024 / layout.dtype.size());
    ASSERT_EQ(6u, image_width(layout));
    ASSERT_EQ(1024u, image_row_pitch(layout));
    ASSERT_EQ(5u, image_height(layout));

    layout = {{5, 3, 1024}, dtype::Float32{}, fmt};
    ASSERT_EQ(layout.stride[2], 1);
    ASSERT_EQ(layout.stride[1], 1024);
    ASSERT_EQ(layout.stride[0], 1024 * 3);
    ASSERT_EQ(1024u * 3u / 4u, image_width(layout));
    ASSERT_EQ(1024u * 3u * 4u, image_row_pitch(layout));
    ASSERT_EQ(5u, image_height(layout));

    fmt = Image2DPack4TensorFormat::make_raw(2, 1024);
    layout = {{3, 5, 8}, dtype::Float32{}, fmt};
    ASSERT_EQ(layout.stride[2], 1);
    ASSERT_EQ(layout.stride[1], 1024 / sizeof(float));
    ASSERT_EQ(layout.stride[0], 1024 / sizeof(float) * 5);
    ASSERT_TRUE(layout.is_contiguous());
    {
        auto&& impl = fmt.as_impl<Image2DPack4TensorFormat>();
        ASSERT_EQ(15u, impl.image_height(layout));
        ASSERT_EQ(8u, impl.image_width_elems(layout));
        ASSERT_EQ(2u, impl.image_width(layout));
        ASSERT_EQ(1024u, impl.image_row_pitch(layout));
        ASSERT_EQ(make_layout({15, 8}, {1024 / 4, 1}, layout.dtype),
                  layout.collapse_contiguous());

        // broadcast
        layout.stride[0] = layout.stride[1] = 0;
        ASSERT_EQ(1u, impl.image_height(layout));
        ASSERT_EQ(8u, impl.image_width_elems(layout));
        ASSERT_EQ(2u, impl.image_width(layout));
        ASSERT_EQ(1024u, impl.image_row_pitch(layout));
        ASSERT_EQ(make_layout({15, 8}, {0, 1}, layout.dtype),
                  layout.collapse_contiguous());
    }

    layout = {{3, 4, 1024}, dtype::Float32{}, fmt};
    ASSERT_EQ(layout.stride[2], 1);
    ASSERT_EQ(layout.stride[1], 1024);
    ASSERT_EQ(layout.stride[0], 1024 * 4);

    fmt = Image2DPack4TensorFormat::make_raw(2, 64);
    layout = {{1, 1, 1, 2, 4}, dtype::Float16{}, fmt};
    {
        auto contig = layout.collapse_contiguous();
        auto&& impl = contig.format.as_impl<Image2DPack4TensorFormat>();
        ASSERT_EQ(make_layout({1, 8}, {32, 1}, layout.dtype), contig);
        ASSERT_EQ(1u, impl.align_axis());
        ASSERT_EQ(64u, impl.align_size_in_byte());
    }
}

TEST(BASIC_TYPES, TENSOR_LAYOUT_FMT_SMALL_ALLIGN) {
    TensorFormat fmt = Image2DPack4TensorFormat::make_raw(1, 1);
    TensorLayout layout{{3, 5, 8}, dtype::Float32{}, fmt};
    ASSERT_EQ(layout.stride[2], 1);
    ASSERT_EQ(layout.stride[1], 8);
    ASSERT_EQ(layout.stride[0], 40);
    ASSERT_EQ(10u, image_width(layout));
    ASSERT_EQ(160u, image_row_pitch(layout));
    ASSERT_EQ(3u, image_height(layout));
}

TEST(BASIC_TYPES, TENSOR_LAYOUT_FMT_COLLAPSE_H) {
    // collapse multiple 1s on height
    auto fmt = Image2DPack4TensorFormat::make_raw(2, 64);
    for (size_t v0 : {1, 3}) {
        TensorLayout layout{{v0, 1, 1, 2, 4}, dtype::Float16{}, fmt};

        auto contig = layout.collapse_contiguous();
        auto&& impl = contig.format.as_impl<Image2DPack4TensorFormat>();
        ASSERT_EQ(make_layout({v0, 8}, {32, 1}, layout.dtype), contig);
        ASSERT_EQ(1u, impl.align_axis());
        ASSERT_EQ(64u, impl.align_size_in_byte());
    }
}

TEST(BASIC_TYPES, TENSOR_LAYOUT_FMT_COLLAPSE_W) {
    // collapse multiple 1s on width
    auto fmt = Image2DPack4TensorFormat::make_raw(2, 64);
    for (size_t v0 : {1, 3}) {
        TensorLayout layout{{1, 2, v0, 10, 4}, dtype::Float16{}, fmt};

        auto contig = layout.collapse_contiguous();
        auto&& impl = contig.format.as_impl<Image2DPack4TensorFormat>();
        ASSERT_EQ(make_layout({2, v0 * 40}, {v0 == 1 ? 64 : 128, 1},
                              layout.dtype),
                  contig);
        ASSERT_EQ(1u, impl.align_axis());
        ASSERT_EQ(64u, impl.align_size_in_byte());
    }
}

// vim: syntax=cpp.doxygen
