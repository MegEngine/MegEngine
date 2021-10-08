/**
 * \file dnn/src/arm_common/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/elemwise/opr_impl.h"
#include "src/arm_common/elemwise/binary/algo.h"
#include "src/arm_common/elemwise/ternary/algo.h"
#include "src/arm_common/elemwise/unary/algo.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/metahelper.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace arm_common;

class ElemwiseImpl::AlgoPack {
    AlgoUnary algo_unary;
    AlgoBinaryVecVec algo_binary_vec_vec;
    AlgoBinaryVecScalar algo_binary_vec_sca;
    AlgoBinaryVecBcast101 algo_binary_vec_bcast101;
    AlgoBinaryVecBcast111C algo_binary_vec_bcast110;
    AlgoBinaryVecBcast101xX algo_binary_VEC_BCAST101xX;
    AlgoTernaryFma3VecVecVec algo_ternaryfma3_vec_vec_vec;
    AlgoTernaryFma3VecVecScalar algo_ternaryfma3_vec_vecsca;
    AlgoTernaryFma3Bcast101VecBcast101 algo_ternaryfma3_bcast101_vec_bcast101;
    AlgoTernaryFma3Bcast111CVecBcast111C algo_ternaryfma3_bcast110_vec_bcast110;
    AlgoTernaryFma3Bcast101xXVecBcast101xX algo_ternaryfma3_bcast101xX_vec_bcast101xX;
    AlgoTernaryFma3VecBcast101Vec algo_ternaryfma3_vec_bcast101_vec;
    AlgoTernaryFma3VecBcast111CVec algo_ternaryfma3_vec_bcast110_vec;
    AlgoTernaryFma3VecBcast101xXVec algo_ternaryfma3_vec_bcast101xX_vec;
    AlgoTernaryFma3VecScalarVec algo_ternaryfma3_vec_sca_vec;
    AlgoTernaryFma3VecScalarScalar algo_ternaryfma3_vec_sca_sca;

public:
    AlgoPack() {
        all_algos.emplace_back(&algo_unary);
        all_algos.emplace_back(&algo_binary_vec_vec);
        all_algos.emplace_back(&algo_binary_vec_sca);
        all_algos.emplace_back(&algo_binary_vec_bcast101);
        all_algos.emplace_back(&algo_binary_vec_bcast110);
        all_algos.emplace_back(&algo_binary_VEC_BCAST101xX);
        all_algos.emplace_back(&algo_ternaryfma3_vec_vec_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_vecsca);
        all_algos.emplace_back(&algo_ternaryfma3_bcast101_vec_bcast101);
        all_algos.emplace_back(&algo_ternaryfma3_bcast110_vec_bcast110);
        all_algos.emplace_back(&algo_ternaryfma3_bcast101xX_vec_bcast101xX);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast101_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast110_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_bcast101xX_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_sca_vec);
        all_algos.emplace_back(&algo_ternaryfma3_vec_sca_sca);
    }
    SmallVector<AlgoBase*> all_algos;
};

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    m_src = &srcs;
    m_dst = &dst;

    if (!dst.layout.is_contiguous()) {
        return fallback::ElemwiseImpl::exec(srcs, dst);
    }

    if (m_dst->layout.dtype == dtype::Float32() ||
        DNN_FLOAT16_SELECT(m_dst->layout.dtype == dtype::Float16(), false) ||
        m_dst->layout.dtype == dtype::Int32() ||
        m_dst->layout.dtype == dtype::Int16() || m_dst->layout.dtype == dtype::Int8()) {
        auto kern_param = make_kern_param(this);
        kern_param.m_dst = &dst;
        static AlgoPack m_algo_pack;
        for (auto& m_algo : m_algo_pack.all_algos) {
            if (m_algo->is_available(kern_param)) {
                m_algo->exec(kern_param);
                return;
            }
        }
    }
    fallback::ElemwiseImpl::exec(srcs, dst);
}

ElemwiseImpl::KernParam ElemwiseImpl::make_kern_param(ElemwiseImpl* opr) {
    KernParam kern_param;
    kern_param.broad_cast_type = BcastType::UNKNOWN_BCAST_TYPE;
    kern_param.mode = opr->param().mode;
    kern_param.handle = opr->handle();

    auto is_legal_layout_for_nhwc = [](const TensorLayout& l) {
        if (is_vector(l))
            return true;
        if (l.ndim == 2 && l.stride[1] == 1)
            return true;
        return false;
    };

    if ((opr->m_src->size() == 3) && (opr->param().mode == Mode::FUSE_MUL_ADD3)) {
        kern_param.ternary_elparam = opr->make_elemwise_op_param<3>();
        bool c_is_scalar;
        opr->prepare_fma3(kern_param.ternary_elparam, c_is_scalar);
        auto &src0 = kern_param.ternary_elparam[0],
             &src1 = kern_param.ternary_elparam[1],
             &src2 = kern_param.ternary_elparam[2];
        BroadcastChannelInfo binfo;

        if (is_vector(src0.layout) && is_vector(src1.layout) &&
            is_vector(src2.layout)) {
            kern_param.broad_cast_type = BcastType::VEC_VEC_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && is_vector(src1.layout) && c_is_scalar) {
            kern_param.broad_cast_type = BcastType::VEC_VEC_SCALAR;
            return kern_param;
        }

        if (is_vector(src1.layout) && is_broadcasted_channel_like(src0.layout, binfo) &&
            src0.layout.eq_layout(src2.layout)) {
            kern_param.broad_cast_type = BcastType::BCAST101_VEC_BCAST101;
            return kern_param;
        }

        if (is_vector(src1.layout) &&
            (is_broadcastedx_channel_like<4>(src0.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src0.layout, binfo)) &&
            src0.layout.eq_layout(src2.layout)) {
            kern_param.broad_cast_type = BcastType::BCAST101xX_VEC_BCAST101xX;
            return kern_param;
        }

        if (is_vector(src0.layout) && src0.layout.eq_layout(src2.layout) &&
            is_broadcasted_channel_like(src1.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST101_VEC;
            return kern_param;
        }

        if (is_legal_layout_for_nhwc(src1.layout) &&
            is_NHWC_broadcasted_channel_like(src0.layout, binfo) &&
            src0.layout.eq_layout(src2.layout)) {
            kern_param.broad_cast_type = BcastType::BCAST111C_VEC_BCAST111C;
            return kern_param;
        }

        if (is_legal_layout_for_nhwc(src0.layout) &&
            src2.layout.eq_layout(src0.layout) &&
            is_NHWC_broadcasted_channel_like(src1.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST111C_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && src0.layout.eq_layout(src2.layout) &&
            (is_broadcastedx_channel_like<4>(src1.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src1.layout, binfo))) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST101xX_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && is_vector(src2.layout) &&
            is_broadcasted_scalar(src1.layout)) {
            kern_param.broad_cast_type = BcastType::VEC_SCALAR_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && is_broadcasted_scalar(src1.layout) &&
            is_broadcasted_scalar(src2.layout)) {
            kern_param.broad_cast_type = BcastType::VEC_SCALAR_SCALAR;
            return kern_param;
        }
    } else if (opr->m_src->size() == 2) {
        kern_param.binary_elparam = opr->make_elemwise_op_param<2>();
        auto &src0 = kern_param.binary_elparam[0], &src1 = kern_param.binary_elparam[1];
        BroadcastChannelInfo binfo;
        if (is_vector(src0.layout) && is_vector(src1.layout)) {
            kern_param.broad_cast_type = BcastType::VEC_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && is_broadcasted_scalar(src1.layout)) {
            kern_param.broad_cast_type = BcastType::VEC_SCALAR;
            return kern_param;
        }

        if (is_vector(src1.layout) && is_broadcasted_scalar(src0.layout)) {
            kern_param.broad_cast_type = BcastType::SCALAR_VEC;
            return kern_param;
        }

        if (is_vector(src0.layout) && is_broadcasted_channel_like(src1.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST101;
            return kern_param;
        }

        if (is_vector(src1.layout) && is_broadcasted_channel_like(src0.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::BCAST101_VEC;
            return kern_param;
        }

        if (is_legal_layout_for_nhwc(src1.layout) &&
            is_NHWC_broadcasted_channel_like(src0.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::BCAST111C_VEC;
            return kern_param;
        }

        if (is_legal_layout_for_nhwc(src0.layout) &&
            is_NHWC_broadcasted_channel_like(src1.layout, binfo)) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST111C;
            return kern_param;
        }

        if (is_vector(src0.layout) &&
            (is_broadcastedx_channel_like<4>(src1.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src1.layout, binfo))) {
            kern_param.broad_cast_type = BcastType::VEC_BCAST101xX;
            return kern_param;
        }

        if (is_vector(src1.layout) &&
            (is_broadcastedx_channel_like<4>(src0.layout, binfo) ||
             is_broadcastedx_channel_like<8>(src0.layout, binfo))) {
            kern_param.broad_cast_type = BcastType::BCAST101xX_VEC;
            return kern_param;
        }
    } else if (opr->m_src->size() == 1) {
        kern_param.broad_cast_type = BcastType::VEC;
        kern_param.unary_elparam = opr->make_elemwise_op_param<1>();
        return kern_param;
    }

    return kern_param;
}

// vim: syntax=cpp.doxygen
