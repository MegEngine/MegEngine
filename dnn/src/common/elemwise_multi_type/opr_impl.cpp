/**
 * \file dnn/src/common/elemwise_multi_type/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <mutex>
#include "megdnn/oprs.h"
#include "src/common/utils.h"

#include "midout.h"
MIDOUT_DECL(megdnn_common_elemwise_multi_type)

using namespace megdnn;

using Mode = ElemwiseMultiType::Mode;
using ModeTrait = ElemwiseMultiType::ModeTrait;

namespace {
void check_dtype(const ModeTrait& trait, size_t i, const TensorLayout& src) {
    trait.check_inp[i](src.dtype);
}
}  // anonymous namespace

const ModeTrait& ModeTrait::from_mode(Mode mode) {
    static std::mutex mtx;
    static std::vector<ModeTrait> traits;

    std::lock_guard<std::mutex> _lock(mtx);

    auto make_check_dtype_func = [](DType expected) {
        auto func = [expected](DType dtype) {
            megdnn_assert(expected.enumv() == dtype.enumv(),
                          "expected %s, but got %s", expected.name(),
                          dtype.name());
        };
        return func;
    };

    auto make_check_category = [](DTypeCategory expected) {
        auto func = [expected](DType dtype) {
            megdnn_assert(expected == dtype.category());
        };
        return func;
    };

    auto make_out_dtype_func = [](DType expected) {
        auto func = [expected](DType& dtype, bool check) {
            if (check) {
                megdnn_assert(expected.enumv() == dtype.enumv(),
                              "expected %s, but got %s", expected.name(),
                              dtype.name());
            } else {
                dtype = expected;
            }
        };
        return func;
    };

    auto make_out_category_func = [](DTypeCategory expected) {
        auto func = [expected](DType& dtype, bool) {
            megdnn_assert(expected == dtype.category());
        };
        return func;
    };

    if (traits.empty()) {
        traits.resize(Param::MODE_NR_MEMBER);
        auto init_fma3_int16x32x32x32 = [&](ModeTrait& dst, const char* name) {
            dst.arity = 3;
            dst.check_inp[0] = make_check_dtype_func(dtype::Int16());
            dst.check_inp[1] = make_check_dtype_func(dtype::Int32());
            dst.check_inp[2] = make_check_dtype_func(dtype::Int32());
            dst.check_out = make_out_dtype_func(dtype::Int32());
            dst.name = name;
        };
        auto init_fma3_iXxf32xf32xi8 = [&](ModeTrait& dst, const char* name) {
            dst.arity = 3;
            dst.check_inp[0] = make_check_category(DTypeCategory::INT);
            dst.check_inp[1] = make_check_dtype_func(dtype::Float32());
            dst.check_inp[2] = make_check_dtype_func(dtype::Float32());
            dst.check_out = make_out_dtype_func(dtype::Int8());
            dst.name = name;
        };
        auto init_rshrs_iXxi8xi8 = [&](ModeTrait& dst, const char* name) {
            dst.arity = 2;
            dst.check_inp[0] = make_check_category(DTypeCategory::INT);
            dst.check_inp[1] = make_check_dtype_func(dtype::Int8());
            dst.check_out = make_out_dtype_func(dtype::Int8());
            dst.name = name;
        };
        auto init_fuse_add_rmulh_rshr_int16x16x16x8 = [&](ModeTrait& dst,
                                                          const char* name) {
            // TODO: This is stupid, we should parameterize shift
            //                   offset, minv and maxv.
            dst.arity = 6;

            dst.check_inp[0] = make_check_dtype_func(dtype::Int16());
            dst.check_inp[1] = make_check_dtype_func(dtype::Int16());
            dst.check_inp[2] = make_check_dtype_func(dtype::Int16());
            dst.check_inp[3] = make_check_dtype_func(dtype::Int8());
            dst.check_inp[4] = make_check_dtype_func(dtype::Int8());
            dst.check_inp[5] = make_check_dtype_func(dtype::Int8());
            dst.check_out = make_out_dtype_func(dtype::Int8());
            dst.name = name;
        };
        auto init_fuse_add_rmulh_rshr_int32x32x32x8 = [&](ModeTrait& dst,
                                                          const char* name) {
            dst.arity = 6;
            dst.check_inp[0] = make_check_dtype_func(dtype::Int32());
            dst.check_inp[1] = make_check_dtype_func(dtype::Int32());
            dst.check_inp[2] = make_check_dtype_func(dtype::Int32());
            dst.check_inp[3] = make_check_dtype_func(dtype::Int8());
            dst.check_inp[4] = make_check_dtype_func(dtype::Int8());
            dst.check_inp[5] = make_check_dtype_func(dtype::Int8());
            dst.check_out = make_out_dtype_func(dtype::Int8());
            dst.name = name;
        };
        auto init_rshrs_iXxi8xi16 = [&](ModeTrait& dst, const char* name) {
            dst.arity = 2;
            dst.check_inp[0] = make_check_category(DTypeCategory::INT);
            dst.check_inp[1] = make_check_dtype_func(dtype::Int8());
            dst.check_out = make_out_dtype_func(dtype::Int16());
            dst.name = name;
        };

        auto init_quantized_unary_op = [&](ModeTrait& dst, const char* name) {
            dst.arity = 1;
            dst.check_inp[0] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_out = make_out_category_func(DTypeCategory::QUANTIZED);
            dst.name = name;
            dst.need_specify_out_dtype = true;
        };

        auto init_quantized_binary_op = [&](ModeTrait& dst, const char* name) {
            dst.arity = 2;
            dst.check_inp[0] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_inp[1] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_out = make_out_category_func(DTypeCategory::QUANTIZED);
            dst.name = name;
            dst.need_specify_out_dtype = true;
        };

        auto init_quantized_ternary_op = [&](ModeTrait& dst, const char* name) {
            dst.arity = 3;
            dst.check_inp[0] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_inp[1] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_inp[2] = make_check_category(DTypeCategory::QUANTIZED);
            dst.check_out = make_out_category_func(DTypeCategory::QUANTIZED);
            dst.name = name;
            dst.need_specify_out_dtype = true;
        };

#define SET(f, m)                                                         \
    MIDOUT_BEGIN(megdnn_common_elemwise_multi_type, midout_iv(Mode::m)) { \
        f(traits[static_cast<int>(Mode::m)], megdnn_mangle(#m));          \
    }                                                                     \
    MIDOUT_END();
        SET(init_fma3_int16x32x32x32, FUSE_MUL_ADD3_INT16x32x32x32);
        SET(init_fma3_iXxf32xf32xi8, FUSE_MUL_ADD3_IXxF32xF32xI8);
        SET(init_rshrs_iXxi8xi8, ROUND_SHR_SATURATE_IXxI8xI8);
        SET(init_fuse_add_rmulh_rshr_int16x16x16x8,
            FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT16x16x16x8);
        SET(init_fuse_add_rmulh_rshr_int32x32x32x8,
            FUSE_ADD_RMULH_ROUND_SHR_SATURATE_INT32x32x32x8);
        SET(init_rshrs_iXxi8xi16, ROUND_SHR_SATURATE_IXxI8xI16);

        //! quantized opr, with specified dtype.
        //! dispatch elemwise mode internally
        SET(init_quantized_unary_op, QRELU);
        SET(init_quantized_unary_op, QABS);
        SET(init_quantized_unary_op, QACOS);
        SET(init_quantized_unary_op, QASIN);
        SET(init_quantized_unary_op, QCEIL);
        SET(init_quantized_unary_op, QCOS);
        SET(init_quantized_unary_op, QEXP);
        SET(init_quantized_unary_op, QEXPM1);
        SET(init_quantized_unary_op, QFLOOR);
        SET(init_quantized_unary_op, QLOG);
        SET(init_quantized_unary_op, QLOG1P);
        SET(init_quantized_unary_op, QNEGATE);
        SET(init_quantized_unary_op, QSIGMOID);
        SET(init_quantized_unary_op, QSIN);
        SET(init_quantized_unary_op, QTANH);
        SET(init_quantized_unary_op, QFAST_TANH);
        SET(init_quantized_unary_op, QROUND);
        SET(init_quantized_unary_op, QERF);
        SET(init_quantized_unary_op, QERFINV);
        SET(init_quantized_unary_op, QERFC);
        SET(init_quantized_unary_op, QERFCINV);
        SET(init_quantized_unary_op, QH_SWISH);

        SET(init_quantized_binary_op, QABS_GRAD);
        SET(init_quantized_binary_op, QADD);
        SET(init_quantized_binary_op, QFLOOR_DIV);
        SET(init_quantized_binary_op, QMAX);
        SET(init_quantized_binary_op, QMIN);
        SET(init_quantized_binary_op, QMOD);
        SET(init_quantized_binary_op, QMUL);
        SET(init_quantized_binary_op, QPOW);
        SET(init_quantized_binary_op, QSIGMOID_GRAD);
        SET(init_quantized_binary_op, QSUB);
        SET(init_quantized_binary_op, QSWITCH_GT0);
        SET(init_quantized_binary_op, QTANH_GRAD);
        SET(init_quantized_binary_op, QTRUE_DIV);
        SET(init_quantized_binary_op, QLOG_SUM_EXP);

        SET(init_quantized_binary_op, QLT);
        SET(init_quantized_binary_op, QLEQ);
        SET(init_quantized_binary_op, QEQ);

        SET(init_quantized_binary_op, QFUSE_ADD_RELU);
        SET(init_quantized_binary_op, QFUSE_ADD_SIGMOID);
        SET(init_quantized_binary_op, QFUSE_ADD_TANH);
        SET(init_quantized_binary_op, QFAST_TANH_GRAD);
        SET(init_quantized_binary_op, QATAN2);
        SET(init_quantized_binary_op, QH_SWISH_GRAD);
        SET(init_quantized_binary_op, QFUSE_ADD_H_SWISH);

        SET(init_quantized_ternary_op, QFUSE_MUL_ADD3);
        SET(init_quantized_ternary_op, QCOND_LEQ_MOV);
#undef SET
    }

    return traits.at(static_cast<int>(mode));
}

void ElemwiseMultiType::deduce_layout(const TensorLayoutArray& src,
                                      TensorLayout& dst) {
    auto trait = mode_trait();
    megdnn_assert(src.size() == trait.arity);
    for (size_t i = 0; i < trait.arity; ++i) {
        check_dtype(trait, i, src[i]);
    }
    TensorShapeArray src_shp;
    for (auto&& i : src)
        src_shp.push_back(i);
    Elemwise::deduce_shape(src_shp, dst);
    dst.init_contiguous_stride();
    trait.check_out(dst.dtype, false);
}

void ElemwiseMultiType::check_layout_and_broadcast(
        const TensorLayoutPtrArray& src, const TensorLayout& dst) {
    auto trait = mode_trait();
    megdnn_assert(src.size() == trait.arity);
    for (size_t i = 0; i < trait.arity; ++i) {
        check_dtype(trait, i, *src[i]);
        *src[i] = src[i]->broadcast(dst);
    }
    auto dtype = dst.dtype;
    trait.check_out(dtype, true);
    megdnn_assert(dst.is_contiguous());
}

// vim: syntax=cpp.doxygen
