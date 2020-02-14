/**
 * \file dnn/test/common/elemwise_multi_type.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/elemwise_multi_type.h"
#include "test/common/checker.h"
#include "test/common/utils.h"

#include "megdnn/oprs/general.h"

using namespace megdnn;
using namespace test;

namespace {

void fuse_add_rmulh_round_shr_saturate_extra_opr_impl(
        const TensorNDArray& data) {
    megdnn_assert(data.size() == 7);
    auto handle = create_cpu_handle(2);
    auto elem_opr = handle->create_operator<Elemwise>();
    auto elem_mt_opr = handle->create_operator<ElemwiseMultiType>();

    size_t st_bytes = data[0].layout.span().dist_byte();
    size_t out_bytes = data[6].layout.span().dist_byte();
    std::vector<uint8_t> tmp_storage(st_bytes), tmq_storage(st_bytes),
        outlike_storage(out_bytes),
        outlike2_storage(out_bytes);
    TensorND tmp{tmp_storage.data(), data[0].layout},
            tmq{tmq_storage.data(), data[0].layout},
            outlike{outlike_storage.data(), data[6].layout},
            outlike2{outlike2_storage.data(), data[6].layout};
    tmp.layout.init_contiguous_stride();
    tmq.layout.init_contiguous_stride();
    outlike.layout.init_contiguous_stride();
    outlike2.layout.init_contiguous_stride();

    elem_opr->param().mode = Elemwise::Mode::ADD;
    elem_opr->exec({data[0], data[1]}, tmp);
    elem_opr->param().mode = Elemwise::Mode::RMULH;
    elem_opr->exec({tmp, data[2]}, tmq);

    elem_mt_opr->param().mode =
            ElemwiseMultiType::Mode::ROUND_SHR_SATURATE_IXxI8xI8;
    elem_mt_opr->exec({tmq, data[3]}, outlike);
    
    elem_opr->param().mode = Elemwise::Mode::MAX;
    elem_opr->exec({outlike, data[4]}, outlike2);

    elem_opr->param().mode = Elemwise::Mode::MIN;
    elem_opr->exec({outlike2, data[5]}, data[6]);
}

}  // namespace

namespace megdnn {
namespace test {
namespace elemwise_multi_type {

#define DEF_TEST(name) \
    template <>        \
    void run_test<name>(Handle * handle)

DEF_TEST(fuse_mul_add3_int16x32x32x32) {
    Checker<ElemwiseMultiType> checker(handle);
    checker.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32});
    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    UniformIntRNG rng{-100, 100};
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.execs({{5, 6, 7}, {1, 6, 1}, {1, 6, 1}, {}})
            .execs({{1, 600, 700}, {1, 600, 1}, {1, 600, 1}, {}})
            .execs({{102, 67, 71}, {1, 67, 1}, {1, 67, 1}, {}});
}

DEF_TEST(fuse_mul_add3_iXxf32xf32xi8) {
    Checker<ElemwiseMultiType> checker(handle);
    checker.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8});
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    std::array<DType, 3> src_types{
            {dtype::Int8{}, dtype::Int16{}, dtype::Int32{}}};
    UniformIntRNG rng{-100, 100};
    checker.set_rng(0, &rng);
    for (DType stype : src_types) {
        checker.set_dtype(0, stype);
        checker.execs({{100, 159}, {100, 159}, {100, 159}, {}})
                .execs({{100, 159}, {1, 159}, {1, 159}, {}})
                .execs({{100, 160}, {100, 160}, {100, 160}, {}})
                .execs({{100, 160}, {1, 160}, {1, 160}, {}});
    }
}

DEF_TEST(round_shr_saturate_iXxi8xi8) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle);

    checker.set_param({Mode::ROUND_SHR_SATURATE_IXxI8xI8})
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int8());

    typedef std::tuple<DType, int, int> TestDesciption;
    std::array<TestDesciption, 3> testcases{
            {TestDesciption{dtype::Int8(), 1, 7},
             TestDesciption{dtype::Int16(), 2, 14},
             TestDesciption{dtype::Int32(), 7, 25}}};
    for (auto desc : testcases) {
        DType dtype;
        int l, r;
        std::tie(dtype, l, r) = desc;

        UniformIntRNG rng{l, r};
        checker.set_dtype(0, dtype)
                .set_rng(1, &rng)
                .execs({{7, 9, 11, 13}, {1}, {}})
                .execs({{16, 3, 256, 256}, {1}, {}})
                .execs({{3, 7, 11}, {1}, {}})
                .execs({{5, 23}, {1}, {}})
                .execs({{998}, {1}, {}});
        for (int i = 0; i < 100; i++) {
            checker.execs({{7}, {1}, {}});
        }
    }
}

DEF_TEST(fuse_add_rmulh_round_shr_saturate_int16) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle);

    UniformIntRNG inp_rng{INT16_MIN >> 1, INT16_MAX >> 1};
    UniformIntRNG bias_rng{INT16_MIN >> 1, INT16_MAX >> 1};
    UniformIntRNG offset_rng{2, 14};
    UniformIntRNG minv_rng{-128, -64};
    UniformIntRNG maxv_rng{63, 127};
    checker.set_param({Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8})
            .set_dtype(0, dtype::Int16())
            .set_dtype(1, dtype::Int16())
            .set_dtype(2, dtype::Int16())
            .set_dtype(3, dtype::Int8())
            .set_dtype(4, dtype::Int8())
            .set_dtype(5, dtype::Int8())
            .set_rng(0, &inp_rng)
            .set_rng(1, &bias_rng)
            .set_rng(2, &inp_rng)
            .set_rng(3, &offset_rng)
            .set_rng(4, &minv_rng)
            .set_rng(5, &maxv_rng)
            .set_extra_opr_impl(
                    fuse_add_rmulh_round_shr_saturate_extra_opr_impl);
    auto run_with_shape = [&](const TensorShape& shape0,
                              const TensorShape& shape1) {
        checker.execs({shape0, shape1, {1}, {1}, {1}, {1}, {}});
    };
    run_with_shape({7, 9, 11, 13}, {1, 9, 1, 1});
    run_with_shape({16, 3, 256, 256}, {1, 3, 1, 1});
}

DEF_TEST(fuse_add_rmulh_round_shr_saturate_int32) {
    using Mode = ElemwiseMultiType::Param::Mode;
    Checker<ElemwiseMultiType> checker(handle);

    UniformIntRNG inp_rng{INT32_MIN >> 1, INT32_MAX >> 1};
    UniformIntRNG bias_rng{INT32_MIN >> 1, INT32_MAX >> 1};
    UniformIntRNG offset_rng{7, 25};
    UniformIntRNG minv_rng{-128, -64};
    UniformIntRNG maxv_rng{63, 127};
    checker.set_param({Mode::FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8})
            .set_dtype(0, dtype::Int32())
            .set_dtype(1, dtype::Int32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int8())
            .set_dtype(4, dtype::Int8())
            .set_dtype(5, dtype::Int8())
            .set_rng(0, &inp_rng)
            .set_rng(1, &bias_rng)
            .set_rng(2, &inp_rng)
            .set_rng(3, &offset_rng)
            .set_rng(4, &minv_rng)
            .set_rng(5, &maxv_rng)
            .set_extra_opr_impl(
                    fuse_add_rmulh_round_shr_saturate_extra_opr_impl);
    auto run_with_shape = [&](const TensorShape& shape0,
                              const TensorShape& shape1) {
        checker.execs({shape0, shape1, {1}, {1}, {1}, {1}, {}});
    };
    run_with_shape({7, 9, 11, 13}, {1, 9, 1, 1});
    run_with_shape({16, 3, 256, 256}, {1, 3, 1, 1});
}

}  // namespace elemwise_multi_type
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
