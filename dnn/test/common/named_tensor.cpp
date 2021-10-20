/**
 * \file dnn/test/common/named_tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/named_tensor.h"

// clang-format off
#include "test/common/utils.h"
#include "test/common/fix_gtest_on_platforms_without_exception.inl"
// clang-format on

using namespace megdnn;
using megdnn::test::MegDNNError;

TEST(NAMED_TENSOR, NAMED_TENSOR_SHAPE_BASIC) {
    ASSERT_EQ(Dimension::NR_NAMES, 10);
    Dimension dim0 = {"C"}, dim1 = {"C//32"}, dim2 = {"C//4"}, dim3 = {"C%32"},
              dim4 = {"C%4"}, dim5 = {"C//4%8"};
    ASSERT_TRUE(dim0 == dim1 * dim3);
    ASSERT_TRUE(dim2 == dim1 * dim5);
    ASSERT_THROW(dim0 * dim0, MegDNNError);

    ASSERT_TRUE(dim1 == dim0 / dim3);
    ASSERT_TRUE(dim3 == dim0 / dim1);
    ASSERT_TRUE(dim4 == dim3 / dim5);
    ASSERT_TRUE(dim5 == dim3 / dim4);
    ASSERT_TRUE(dim5 == dim2 / dim1);
    ASSERT_THROW(dim5 / dim1, MegDNNError);

    ASSERT_TRUE(dim1 < dim4);
    ASSERT_TRUE(dim5 < dim4);
    ASSERT_FALSE(dim4 < dim5);
    ASSERT_TRUE(dim1 < dim2);
    ASSERT_FALSE(dim2 < dim1);

    auto shape0 =
            NamedTensorShape::make_named_tensor_shape(NamedTensorShape::Format::NCHW);
    SmallVector<Dimension> dims = {{"N"}, {"C"}, {"H"}, {"W"}};
    NamedTensorShape shape1(dims);
    NamedTensorShape shape2{{"N"}, {"C"}, {"H"}, {"W"}};
    ASSERT_TRUE(shape0.eq_shape(shape1));
    ASSERT_TRUE(shape0.eq_shape(shape2));
    ASSERT_TRUE(shape1.eq_shape(shape2));
    auto shape3 =
            NamedTensorShape::make_named_tensor_shape(NamedTensorShape::Format::NCHW4);
    ASSERT_FALSE(shape0.eq_shape(shape3));
    auto shape4 = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW44_DOT);
    std::sort(shape4.dims.begin(), shape4.dims.begin() + shape4.ndim);
    NamedTensorShape shape5{{"N"}, {"C//32"}, {"H"}, {"W"}, {"C//8%4"}, {"C%8"}};
    std::sort(shape5.dims.begin(), shape5.dims.begin() + shape5.ndim);
}
// vim: syntax=cpp.doxygen
