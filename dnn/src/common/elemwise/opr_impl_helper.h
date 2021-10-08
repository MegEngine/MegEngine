/**
 * \file dnn/src/common/elemwise/opr_impl_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/oprs/general.h"
#include "megdnn/tensor_format.h"

#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"

namespace megdnn {
class ElemwiseLayoutHelper {
public:
    //! describe broadcasted [1, y, 1] to [x, y, z]
    struct BroadcastChannelInfo {
        size_t x, y, z;

        bool operator==(const BroadcastChannelInfo& rhs) const {
            return x == rhs.x && y == rhs.y && z == rhs.z;
        }
    };

    //! describe broadcasted [1, y] to [x, y]
    struct Broadcast1xInfo {
        size_t x, y;

        bool operator==(const Broadcast1xInfo& rhs) const {
            return x == rhs.x && y == rhs.y;
        }
    };

    /*!
     * \brief check layout and get canonized op param
     * \param opr operator pointer
     * \param check_layout_and_broadcast function pointer to implement
     *      check_layout_and_broadcast(); operator pointer would be passed
     *      to it
     */
    template <int arity>
    static ElemwiseOpParamN<arity> make_elemwise_op_param(
            void* opr,
            void (*check_layout_and_broadcast)(
                    void*, const TensorLayoutPtrArray&, const TensorLayout&),
            const TensorNDArray& src, const TensorND& dst);

    //! check whether given layout is 1D contig
    static bool is_vector(const TensorLayout& layout) {
        if (layout.format.type() != TensorFormat::Type::DEFAULT) {
            return layout.is_contiguous();
        }
        return layout.ndim == 1 && layout.stride[0] == 1;
    }

    /*!
     * \brief check whether it is compatible with (1, x) broadcasted into (y, x)
     *
     * Note: input can be one-dimensional.
     */
    static bool is_broadcasted_1x(const TensorLayout& layout, Broadcast1xInfo& binfo);

    //! check whether given layout is broadcasted scalar
    static bool is_broadcasted_scalar(const TensorLayout& layout);

    /*!
     * \brief check whether layout matches BroadcastChannelInfo
     *
     * Note that Input can also be 2-dimensional, and must be [y, 1] broadacsted
     * into [y, z]; in such case x would be set to 1.
     */
    static bool is_broadcasted_channel_like(
            const TensorLayout& layout, BroadcastChannelInfo& info);

    /*!
     * \brief check whether layout matches BroadcastChannelInfo under NHWC
     * layout
     *
     * Note that Input must be 2-dimensional, and must be [1, y] broadacsted
     * into [z, y] and x would be set to 1.
     */
    static bool is_NHWC_broadcasted_channel_like(
            const TensorLayout& layout, BroadcastChannelInfo& info);

    /*!
     * \brief check whether layout matches BroadcastChannelInfo
     *
     * Note that Input can also be 3-dimensional, and must be [x, 1, z]
     * broadacsted into [x, y, z]
     */
    template <size_t slice_size>
    static bool is_broadcastedx_channel_like(
            const TensorLayout& layout, BroadcastChannelInfo& info);
};

class ElemwiseForwardImplHelper : public ElemwiseForward,
                                  protected ElemwiseLayoutHelper {
    static void call_check_layout_and_broadcast(
            void* opr, const TensorLayoutPtrArray& src, const TensorLayout& dst) {
        return static_cast<ElemwiseForwardImplHelper*>(opr)->check_layout_and_broadcast(
                src, dst);
    }

protected:
    const TensorNDArray* m_src = nullptr;
    const TensorND* m_dst = nullptr;

    /*!
     * \brief check layout and get canonized op param
     *
     * Require that m_src and m_dst have been setup
     */
    template <int arity>
    ElemwiseOpParamN<arity> make_elemwise_op_param() {
        return ElemwiseLayoutHelper::make_elemwise_op_param<arity>(
                this, call_check_layout_and_broadcast, *m_src, *m_dst);
    }

    /*!
     * \brief canonize params for FMA3
     * \param[out] c_is_scalar if true, params[2] has same layout as
     *     params[0]; otherwise params[2] is scalar
     */
    void prepare_fma3(ElemwiseOpParamN<3>& param, bool& c_is_scalar);

    /*!
     * \brief canonize params for FMA4
     * \param[out] guaranteed that params[2] has same layout as
     *      params[0], and params[3] same with params[1].
     */
    void prepare_fma4(ElemwiseOpParamN<4>& param);

public:
    using ElemwiseForward::ElemwiseForward;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
