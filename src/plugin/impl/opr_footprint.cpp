/**
 * \file src/plugin/impl/opr_footprint.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#if MGB_ENABLE_JSON
#include "megdnn/opr_param_json.h"
#endif

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_opr_footprint)
#define MIDOUT_B(...) \
    MIDOUT_BEGIN(megbrain_opr_footprint, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;

namespace {

template <class T>
uint64_t opr_footprint_func(cg::OperatorNodeBase* opr);

// Elemwise
template <>
uint64_t opr_footprint_func<opr::Elemwise>(cg::OperatorNodeBase* opr) {
    return opr->output()[0]->shape().total_nr_elems() *
           (std::max<size_t>(opr->input().size(), 2) - 1);
}

// AddUpdate
template <>
uint64_t opr_footprint_func<opr::AddUpdate>(cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2,
               "AddUpdate opr should have two inputs");
    auto&& out_shape = opr->output()[0]->shape();
    return out_shape.total_nr_elems() * 3;
}

template <class Conv>
uint64_t eval_conv_computation(const TensorShape& src_shape,
                               const TensorShape& filter_shape,
                               const TensorShape& dst_shape,
                               cg::OperatorNodeBase* opr) {
    using Param = opr::ConvolutionForward::Param;
    auto&& param = opr->cast_final_safe<Conv>().param();

    if (param.format == Param::Format::NHWCD4) {
        size_t fh, fw;
        size_t group = 1;
        if (param.sparse == Param::Sparse::DENSE) {
            fh = filter_shape[1];
            fw = filter_shape[2];
            group = 1;
        } else {
            // chanwise conv
            mgb_assert(param.sparse == Param::Sparse::GROUP);
            fh = filter_shape[2];
            fw = filter_shape[3];
            group = filter_shape[0];

            if (filter_shape.ndim == 5) {
                group *= 4;
            }
        }
        return dst_shape.total_nr_elems() * fh * fw *
            src_shape[2] * 4 / group * 2;
    }
    auto eval_conv_computation_nchwx = [&param, &src_shape, &filter_shape,
                                        &dst_shape]() -> uint64_t {
        size_t fh, fw;
        bool hybird_nchwx = false;
        size_t group = 1;
        if (param.sparse == Param::Sparse::DENSE) {
            //! if nchwxx mode src is nchw output is nchwxx
            if (dst_shape.ndim == 5 && src_shape.ndim == 4) {
                fh = filter_shape[1];
                fw = filter_shape[2];
                hybird_nchwx = true;
            } else {
                fh = filter_shape[2];
                fw = filter_shape[3];
            }
            group = 1;
        } else {
            mgb_assert(param.sparse == Param::Sparse::GROUP);
            fh = filter_shape[3];
            fw = filter_shape[4];
            group = filter_shape[0];
        }
        if (param.format == Param::Format::NCHW88) {
            //! if channel wise weight layout is {group/8, FH, FW, 1, 1, 8}
            if (filter_shape[1] == 1 && filter_shape[2] == 1) {
                group *= 8;
            }
            size_t computation = dst_shape.total_nr_elems() * fh * fw *
                                 src_shape[1] / group * 2;
            return hybird_nchwx ? computation : computation * 8;
        }
        if (param.format == Param::Format::NCHW44 ||
            param.format == Param::Format::NCHW44_DOT) {
            //! if channel wise weight layout is {group/4, FH, FW, 1, 1, 4}
            if (filter_shape[1] == 1 && filter_shape[2] == 1) {
                group *= 4;
            }
            size_t computation = dst_shape.total_nr_elems() * fh * fw *
                                 src_shape[1] / group * 2;
            return hybird_nchwx ? computation : computation * 4;
        }
        if (param.format == Param::Format::NCHW32) {
            return dst_shape.total_nr_elems() * fh * fw * src_shape[1] * 32 /
                   group * 2;
        }
        mgb_assert(param.format == Param::Format::NCHW4 ||
                           param.format == Param::Format::NCHW4_NCHW ||
                           param.format == Param::Format::NCHW4_NCHW32,
                   "format should be NCHW4/NCHW4_NCHW/NCHW4_NCHW32");
        return dst_shape.total_nr_elems() * fh * fw * src_shape[1] * 4 / group *
               2;
    };
    auto eval_conv_computation_chwn4 = [&param, &src_shape, &filter_shape,
                                        &dst_shape]() -> uint64_t {
        size_t fh, fw;
        size_t group = 1;
        if (param.sparse == Param::Sparse::DENSE) {
            fh = filter_shape[1];
            fw = filter_shape[2];
            group = 1;
        } else {
            mgb_assert(param.sparse == Param::Sparse::GROUP);
            fh = filter_shape[2];
            fw = filter_shape[3];
            group = filter_shape[0];
        }
        return dst_shape.total_nr_elems() * fh * fw * src_shape[0] * 4 / group *
               2;
    };
    if (param.format == Param::Format::NCHW4 ||
        param.format == Param::Format::NCHW4_NCHW ||
        param.format == Param::Format::NCHW4_NCHW32 || 
        param.format == Param::Format::NCHW88 ||
        param.format == Param::Format::NCHW44 ||
        param.format == Param::Format::NCHW44_DOT ||
        param.format == Param::Format::NCHW32) {
        return eval_conv_computation_nchwx();
    }
    if (param.format == Param::Format::CHWN4) {
        return eval_conv_computation_chwn4();
    }
    size_t cpos;
    size_t spatial_start;
    size_t group = 1;
    switch (param.format) {
        case Param::Format::NCHW:
            cpos = 1;
            spatial_start = 2;
            break;
        case Param::Format::NHWC:
            cpos = 3;
            spatial_start = 1;
            break;
        default:
            mgb_assert(false, "Unknown CONV Param::Format type");
    }
    switch (param.sparse) {
        case Param::Sparse::DENSE:
            mgb_assert(filter_shape.ndim == 4 || filter_shape.ndim == 6,
                       "DENSE conv filter shape dimension should be "
                       "4/6(winograd mk4)");
            break;
        case Param::Sparse::GROUP:
            mgb_assert(filter_shape.ndim == 5 || filter_shape.ndim == 7,
                       "GROUP conv filter shape dimension should be "
                       "5/7(winograd mk4)");
            spatial_start++;
            group = filter_shape[0];
            break;
        default:
            mgb_assert(false, "Unkown CONV Param::Sparse type");
    }

    uint64_t fh = static_cast<uint64_t>(filter_shape[spatial_start]);
    uint64_t fw = static_cast<uint64_t>(filter_shape[spatial_start + 1]);
    
    // mul and add are counted as 2 operations
    
    return dst_shape.total_nr_elems() * fh * fw *
           static_cast<uint64_t>(src_shape[cpos]) / group * 2;
}

// ConvolutionForward
template <>
uint64_t opr_footprint_func<opr::ConvolutionForward>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2,
               "ConvolutionFwd opr should have two inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& src_shape = opr->input()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    return eval_conv_computation<opr::ConvolutionForward>(
            src_shape, filter_shape, out_shape, opr);
}
template <>
uint64_t opr_footprint_func<opr::ConvBiasForward>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2 || opr->input().size() == 3 ||
                       opr->input().size() == 4,
               "ConvBiasForward opr should have two/three/four inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& src_shape = opr->input()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    uint64_t res = eval_conv_computation<opr::ConvBiasForward>(
            src_shape, filter_shape, out_shape, opr);
    if (opr->input().size() == 3) {
        res += out_shape.total_nr_elems();
    }
    return res;
}

// ConvolutionBackwardData
template <>
uint64_t opr_footprint_func<opr::ConvolutionBackwardData>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2 || opr->input().size() == 3,
               "ConvolutionBackwardData opr should have two or three inputs");
    auto&& filter_shape = opr->input()[0]->shape();
    auto&& diff_shape = opr->input()[1]->shape();
    auto&& grad_shape = opr->output()[0]->shape();
    return eval_conv_computation<opr::ConvolutionBackwardData>(
            grad_shape, filter_shape, diff_shape, opr);
}

// ConvolutionBackwardFilter
template <>
uint64_t opr_footprint_func<opr::ConvolutionBackwardFilter>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 3,
               "ConvolutionBackwardData opr should have three inputs");
    auto&& filter_shape = opr->input()[2]->shape();
    auto&& diff_shape = opr->input()[1]->shape();
    auto&& src_shape = opr->input()[0]->shape();
    return eval_conv_computation<opr::ConvolutionBackwardFilter>(
            src_shape, filter_shape, diff_shape, opr);
}

// MatrixMul
template <>
uint64_t opr_footprint_func<opr::MatrixMul>(cg::OperatorNodeBase* opr) {
    auto&& mopr = opr->cast_final_safe<opr::MatrixMul>();
    auto &&i0 = opr->input(0)->shape(), &&i1 = opr->input(1)->shape();
    mgb_assert(i0.ndim == 2 && i1.ndim == 2);
    auto m = i0[0], k0 = i0[1], k1 = i1[0], n = i1[1];
    if (mopr.param().transposeA) {
        std::swap(m, k0);
    }
    if (mopr.param().transposeB) {
        std::swap(k1, n);
    }
    mgb_assert(k0 == k1);
    // mul and add are counted as 2 operations
    return m * k0 * n * 2;
}

template <>
uint64_t opr_footprint_func<opr::LocalShareForward>(cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2,
               "LocalShare opr should have two inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& src_shape = opr->input()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    using Param = opr::LocalShareForward::Param;
    auto&& param = opr->cast_final_safe<opr::LocalShareForward>().param();
    mgb_assert(param.format == Param::Format::NCHW);
    size_t groups = 1;
    size_t kern_spatial_pos = 3;
    if (param.sparse == Param::Sparse::GROUP) {
        groups = filter_shape[0];
        kern_spatial_pos = 4;
    }
    size_t fh = filter_shape[kern_spatial_pos],
           fw = filter_shape[kern_spatial_pos + 1];
    return out_shape.total_nr_elems() * fh * fw * src_shape[1] * 2 / groups;
}

template <>
uint64_t opr_footprint_func<opr::LocalShareBackwardData>(cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 3,
               "LocalShareBackwardData opr should have three inputs");
    auto&& filter_shape = opr->input()[0]->shape();
    auto&& diff_shape = opr->input()[1]->shape();
    auto&& grad_shape = opr->output()[0]->shape();
    using Param = opr::LocalShareForward::Param;
    auto&& param = opr->cast_final_safe<opr::LocalShareBackwardData>().param();
    mgb_assert(param.format == Param::Format::NCHW);
    size_t groups = 1;
    size_t kern_spatial_pos = 3;
    if (param.sparse == Param::Sparse::GROUP) {
        groups = filter_shape[0];
        kern_spatial_pos = 4;
    }
    size_t fh = filter_shape[kern_spatial_pos],
           fw = filter_shape[kern_spatial_pos + 1];
    return diff_shape.total_nr_elems() * fh * fw * grad_shape[1] * 2 / groups;
}

template <>
uint64_t opr_footprint_func<opr::LocalShareBackwardFilter>(cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 3,
               "LocalShareBackwardFilter opr should have three inputs");
    auto&& src_shape = opr->input()[0]->shape();
    auto&& diff_shape = opr->input()[1]->shape();
    auto&& grad_shape = opr->output()[0]->shape();
    using Param = opr::LocalShareForward::Param;
    auto&& param = opr->cast_final_safe<opr::LocalShareBackwardFilter>().param();
    mgb_assert(param.format == Param::Format::NCHW);
    size_t groups = 1;
    size_t kern_spatial_pos = 3;
    if (param.sparse == Param::Sparse::GROUP) {
        groups = grad_shape[0];
        kern_spatial_pos = 4;
    }
    size_t fh = grad_shape[kern_spatial_pos],
           fw = grad_shape[kern_spatial_pos + 1];
    return diff_shape.total_nr_elems() * fh * fw * src_shape[1] * 2 / groups;
}

template <>
uint64_t opr_footprint_func<opr::DeformableConvForward>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 4,
               "DeformableConvForward opr should have four inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    using Param = opr::DeformableConvForward::Param;
    auto&& param = opr->cast_final_safe<opr::Convolution>().param();
    size_t fh, fw, icpg;
    mgb_assert(param.format == Param::Format::NCHW);
    if (param.sparse == Param::Sparse::GROUP) {
        icpg = filter_shape[2];
        fh = filter_shape[3], fw = filter_shape[4];
    } else {
        icpg = filter_shape[1];
        fh = filter_shape[2], fw = filter_shape[3];
    }
    //! conv(1 mul), mask(1, mul), accumulate(1 add)
    return out_shape.total_nr_elems() * fh * fw * icpg * 3;
}

template <>
uint64_t opr_footprint_func<opr::DeformableConvBackwardFilter>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 5,
               "DeformableConvBackwardFilter opr should have four inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    using Param = opr::DeformableConvBackwardFilter::Param;
    auto&& param = opr->cast_final_safe<opr::Convolution>().param();
    size_t fh, fw, icpg;
    mgb_assert(param.format == Param::Format::NCHW);
    if (param.sparse == Param::Sparse::GROUP) {
        icpg = filter_shape[2];
        fh = filter_shape[3], fw = filter_shape[4];
    } else {
        icpg = filter_shape[1];
        fh = filter_shape[2], fw = filter_shape[3];
    }
    //! deconv(1 mul), mask(1 mul), accumulate(1 add), bilinear(4 add, 4mul,
    //! skip)
    return out_shape.total_nr_elems() * fh * fw * icpg * 3;
}

template <>
uint64_t opr_footprint_func<opr::DeformableConvBackwardData>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 5,
               "DeformableConvBackwardData opr should have four inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    using Param = opr::DeformableConvForward::Param;
    auto&& param = opr->cast_final_safe<opr::Convolution>().param();
    size_t fh, fw, icpg;
    mgb_assert(param.format == Param::Format::NCHW);
    if (param.sparse == Param::Sparse::GROUP) {
        icpg = filter_shape[2];
        fh = filter_shape[3], fw = filter_shape[4];
    } else {
        icpg = filter_shape[1];
        fh = filter_shape[2], fw = filter_shape[3];
    }
    //! deconv(1 mul), mask(1 mul), accumulate(1 add), grad_weight(1 mul, skip),
    //! grad_coord(4mul, 4 add)
    return out_shape.total_nr_elems() * fh * fw * icpg * 12;
}

template <>
uint64_t opr_footprint_func<opr::BatchConvBiasForward>(
        cg::OperatorNodeBase* opr) {
    mgb_assert(opr->input().size() == 2 || opr->input().size() == 3 ||
                       opr->input().size() == 4,
               "BatchConvBias opr should have two/three/four inputs");
    auto&& out_shape = opr->output()[0]->shape();
    auto&& src_shape = opr->input()[0]->shape();
    auto&& filter_shape = opr->input()[1]->shape();
    using Param = opr::BatchConvBiasForward::Param;
    auto&& param = opr->cast_final_safe<opr::BatchConvBiasForward>().param();
    mgb_assert(param.format == Param::Format::NCHW4);
    size_t packed_channels = 4;
    size_t kern_spatial_pos = 3;
    size_t fh = filter_shape[kern_spatial_pos],
           fw = filter_shape[kern_spatial_pos + 1];
    return out_shape.total_nr_elems() * fh * fw * src_shape[1] *
           packed_channels * 2;
}

// Pooling
template <>
uint64_t opr_footprint_func<opr::PoolingForward>(cg::OperatorNodeBase* opr) {
    auto&& param = opr->cast_final_safe<opr::PoolingForward>().param();
    auto area = param.window_h * param.window_w;
    return opr->output(0)->shape().total_nr_elems() * area;
}

// Concat
template <>
uint64_t opr_footprint_func<opr::Concat>(cg::OperatorNodeBase* opr) {
    auto&& out_shape = opr->output()[0]->shape();
    return out_shape.total_nr_elems();
}

// Dimshuffle
template <>
uint64_t opr_footprint_func<opr::Dimshuffle>(cg::OperatorNodeBase* opr) {
    auto&& out = opr->output()[0];
    return out->shape().total_nr_elems();
}

// Reduce
template <>
uint64_t opr_footprint_func<opr::Reduce>(cg::OperatorNodeBase* opr) {
    return opr->input()[0]->shape().total_nr_elems();
}

// Host2DeviceCopy
template <>
uint64_t opr_footprint_func<opr::Host2DeviceCopy>(cg::OperatorNodeBase* opr) {
    auto&& out_shape = opr->output()[0]->shape();
    return out_shape.total_nr_elems();
}

/******************* Registe Param Json Functions *************************/
#if MGB_ENABLE_JSON
template <class T>
std::shared_ptr<json::Value> opr_param_json_func(cg::OperatorNodeBase* opr);

#define REGISTE_PARAM_JSON_FUNC(cls)                            \
    template <>                                                 \
    std::shared_ptr<json::Value> opr_param_json_func<opr::cls>( \
            cg::OperatorNodeBase * opr) {                       \
        return opr::opr_param_to_json(                          \
                opr->cast_final_safe<opr::cls>().param());      \
    }

REGISTE_PARAM_JSON_FUNC(Elemwise)
REGISTE_PARAM_JSON_FUNC(ConvolutionForward)
REGISTE_PARAM_JSON_FUNC(Convolution3D)
REGISTE_PARAM_JSON_FUNC(ConvBiasForward)
REGISTE_PARAM_JSON_FUNC(ConvolutionBackwardData)
REGISTE_PARAM_JSON_FUNC(Convolution3DBackwardData)
REGISTE_PARAM_JSON_FUNC(ConvolutionBackwardFilter)
REGISTE_PARAM_JSON_FUNC(MatrixMul)
REGISTE_PARAM_JSON_FUNC(BatchedMatrixMul)
REGISTE_PARAM_JSON_FUNC(Dot)
REGISTE_PARAM_JSON_FUNC(MatrixInverse)
REGISTE_PARAM_JSON_FUNC(PoolingForward)
REGISTE_PARAM_JSON_FUNC(SVD)
REGISTE_PARAM_JSON_FUNC(MaskConvolution)
REGISTE_PARAM_JSON_FUNC(Images2Neibs)
REGISTE_PARAM_JSON_FUNC(Local)
REGISTE_PARAM_JSON_FUNC(GroupLocal)
REGISTE_PARAM_JSON_FUNC(LRN)
REGISTE_PARAM_JSON_FUNC(Concat)
REGISTE_PARAM_JSON_FUNC(Reduce)
REGISTE_PARAM_JSON_FUNC(LocalShareForward)
REGISTE_PARAM_JSON_FUNC(LocalShareBackwardData)
REGISTE_PARAM_JSON_FUNC(LocalShareBackwardFilter)
REGISTE_PARAM_JSON_FUNC(DeformableConvForward)
REGISTE_PARAM_JSON_FUNC(DeformableConvBackwardFilter)
REGISTE_PARAM_JSON_FUNC(DeformableConvBackwardData)
REGISTE_PARAM_JSON_FUNC(BatchConvBiasForward)

template <>
std::shared_ptr<json::Value> opr_param_json_func<opr::Dimshuffle>(
    cg::OperatorNodeBase * opr) {
        auto param = opr->cast_final_safe<opr::Dimshuffle>().param();

        auto pattern = json::Array::make();
        for (size_t i = 0; i < param.pattern_len; i++)
            pattern->add(json::NumberInt::make(param.pattern[i]));

        return json::Object::make({
            {"ndim", json::NumberInt::make(param.ndim)},
            {"pattern", pattern},
        });
    }

template <>
std::shared_ptr<json::Value> opr_param_json_func<opr::AxisAddRemove>(
    cg::OperatorNodeBase * opr) {
        auto param = opr->cast_final_safe<opr::AxisAddRemove>().param();

        auto desc = json::Array::make();
        for (size_t i = 0; i < param.nr_desc; i++) {
            auto axisdesc = param.desc[i];
            desc->add(
                json::Object::make({
                    {"method", json::NumberInt::make(
                        static_cast<int32_t>(axisdesc.method))},
                    {"axisnum", json::NumberInt::make(axisdesc.axis.get_raw())},
                }));
        }

        return json::Object::make({
            {"nr_desc", json::NumberInt::make(param.nr_desc)},
            {"desc", desc},
        });
    }

template <>
std::shared_ptr<json::Value> opr_param_json_func<opr::Subtensor>(
    cg::OperatorNodeBase * opr) {
        auto desc = json::Array::make();
        auto indices = opr->cast_final_safe<opr::Subtensor>().index_desc();
        for (auto &index : indices){
            desc->add(
                json::Object::make({
                    {"axis", json::NumberInt::make(index.axis.get_raw())},
                    {"begin", json::NumberInt::make(index.begin.node() != nullptr)},
                    {"end", json::NumberInt::make(index.end.node() != nullptr)},
                    {"step", json::NumberInt::make(index.step.node() != nullptr)},
                    {"idx", json::NumberInt::make(index.idx.node() != nullptr)},
                }));
        }

        return desc;
    }
#endif // MGB_ENABLE_JSON

}  // namespace

template <class OprType>
void OprFootprint::add_single_comp_footprint() {
    MIDOUT_B(OprType,
             midout_iv(MGB_HASH_STR("OprFootprint::add_single_comp_footprint")))
    auto&& record = m_type2comp_footprint.emplace(OprType::typeinfo(),
                                                  opr_footprint_func<OprType>);
    mgb_assert(record.second, "duplicate opr typeinfo");
    MIDOUT_E
}

#if MGB_ENABLE_JSON
template <class OprType>
void OprFootprint::add_single_param_json() {
    auto&& record = m_type2param_json.emplace(OprType::typeinfo(),
                                              opr_param_json_func<OprType>);
    mgb_assert(record.second, "duplicate opr typeinfo");
}
#endif

void OprFootprint::init_all_footprints() {
    add_single_comp_footprint<opr::Elemwise>();
    add_single_comp_footprint<opr::AddUpdate>();
    add_single_comp_footprint<opr::ConvolutionForward>();
    add_single_comp_footprint<opr::ConvBiasForward>();
    add_single_comp_footprint<opr::ConvolutionBackwardData>();
    add_single_comp_footprint<opr::ConvolutionBackwardFilter>();
    add_single_comp_footprint<opr::MatrixMul>();
    add_single_comp_footprint<opr::PoolingForward>();
    add_single_comp_footprint<opr::Concat>();
    add_single_comp_footprint<opr::Dimshuffle>();
    add_single_comp_footprint<opr::Reduce>();
    add_single_comp_footprint<opr::Host2DeviceCopy>();
    add_single_comp_footprint<opr::LocalShareForward>();
    add_single_comp_footprint<opr::LocalShareBackwardData>();
    add_single_comp_footprint<opr::LocalShareBackwardFilter>();
    add_single_comp_footprint<opr::DeformableConvForward>();
    add_single_comp_footprint<opr::DeformableConvBackwardFilter>();
    add_single_comp_footprint<opr::DeformableConvBackwardData>();
    add_single_comp_footprint<opr::BatchConvBiasForward>();

#if MGB_ENABLE_JSON
    add_single_param_json<opr::Elemwise>();
    add_single_param_json<opr::ConvolutionForward>();
    add_single_param_json<opr::Convolution3D>();
    add_single_param_json<opr::ConvBiasForward>();
    add_single_param_json<opr::ConvolutionBackwardData>();
    add_single_param_json<opr::Convolution3DBackwardData>();
    add_single_param_json<opr::ConvolutionBackwardFilter>();
    add_single_param_json<opr::MatrixMul>();
    add_single_param_json<opr::BatchedMatrixMul>();
    add_single_param_json<opr::Dot>();
    add_single_param_json<opr::MatrixInverse>();
    add_single_param_json<opr::PoolingForward>();
    add_single_param_json<opr::SVD>();
    add_single_param_json<opr::MaskConvolution>();
    add_single_param_json<opr::Images2Neibs>();
    add_single_param_json<opr::Local>();
    add_single_param_json<opr::GroupLocal>();
    add_single_param_json<opr::LRN>();
    add_single_param_json<opr::Concat>();
    add_single_param_json<opr::Dimshuffle>();
    add_single_param_json<opr::AxisAddRemove>();
    add_single_param_json<opr::Subtensor>();
    add_single_param_json<opr::Reduce>();
    add_single_param_json<opr::LocalShareForward>();
    add_single_param_json<opr::LocalShareBackwardData>();
    add_single_param_json<opr::LocalShareBackwardFilter>();
    add_single_param_json<opr::DeformableConvForward>();
    add_single_param_json<opr::DeformableConvBackwardFilter>();
    add_single_param_json<opr::DeformableConvBackwardData>();
    add_single_param_json<opr::BatchConvBiasForward>();

#endif
}

OprFootprint::Result OprFootprint::calc_footprint(cg::OperatorNodeBase* opr) {
    Result rst;
    auto&& dep_map = opr->node_prop().dep_map();
    for (auto&& inp : opr->input()) {
        if (inp->mem_plan().valid())
            rst.inp_layout.push_back(inp->layout());
        else
            rst.inp_layout.push_back({inp->shape(), inp->dtype()});
        if (cg::OperatorNodeBase::NodeProp::is_device_value_dep(
                    dep_map.at(inp))) {
            rst.memory += inp->dtype().size(inp->shape().total_nr_elems());
        }
    }
    for (auto&& out : opr->output()) {
        if (out->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
            continue;
        rst.out_shape.push_back(out->shape());
        rst.memory += out->dtype().size(out->shape().total_nr_elems());
    }
    rst.computation = get_computation(opr);
#if MGB_ENABLE_JSON
    rst.param = get_param_json(opr);
#endif
    rst.opr_type = opr->dyn_typeinfo();
    return rst;
}

uint64_t OprFootprint::get_computation(cg::OperatorNodeBase* opr) {
    auto comp_trait = m_type2comp_footprint.find(opr->dyn_typeinfo());
    if (comp_trait != m_type2comp_footprint.end()) {
        return (comp_trait->second)(opr);
    }
    return 0;
}

#if MGB_ENABLE_JSON
std::shared_ptr<json::Value> OprFootprint::get_param_json(
        cg::OperatorNodeBase* opr) {
    auto param_trait = m_type2param_json.find(opr->dyn_typeinfo());
    if (param_trait != m_type2param_json.end()) {
        return (param_trait->second)(opr);
    }
    return json::Object::make();
}

std::shared_ptr<json::Value> OprFootprint::Result::to_json() const {
    using namespace json;
    std::shared_ptr<Value> comp;
    if (computation) {
        comp = NumberInt::make(computation);
    } else {
        comp = Null::make();
    }
    auto format_shape_arr = [](const TensorShapeArray& arr) {
        auto ret = Array::make();
        for (auto&& shp : arr) {
            auto cur = Array::make();
            for (size_t i = 0; i < shp.ndim; ++i) {
                cur->add(NumberInt::make(shp[i]));
            }
            ret->add(std::move(cur));
        }
        return ret;
    };
    auto format_layout_arr =
            [](const TensorLayoutArray& arr) -> std::shared_ptr<Value> {
        auto ret = Array::make();
        bool have_non_contig = false;
        for (auto&& item : arr) {
            if (item.is_contiguous()) {
                ret->add(json::Null::make());
            } else {
                have_non_contig = true;
                auto cur = Array::make();
                for (size_t i = 0; i < item.ndim; ++i) {
                    cur->add(NumberInt::make(item.stride[i]));
                }
                ret->add(std::move(cur));
            }
        }
        if (!have_non_contig) {
            ret.reset();
        }
        return ret;
    };

    TensorShapeArray inp_shape;
    for (auto&& i : inp_layout)
        inp_shape.push_back(i);
    auto ret = Object::make({{"computation", std::move(comp)},
                             {"memory", NumberInt::make(memory)},
                             {"in_shapes", format_shape_arr(inp_shape)},
                             {"out_shapes", format_shape_arr(out_shape)},
                             {"param", param}});
    if (auto inp_layout_json = format_layout_arr(inp_layout)) {
        ret->operator[]("in_layouts") = std::move(inp_layout_json);
    }
    return ret;
}

std::shared_ptr<json::Value> OprFootprint::get_opr_fp_graph_exec(
        cg::ComputingGraph& graph, const SymbolVarArray& outputs) {
    OprFootprint m_opr_footprint;
    ComputingGraph::OutputSpec out_spec;
    for (auto i : outputs) {
        out_spec.emplace_back(i, nullptr);
    }
    graph.options().allocate_static_mem_after_graph_compile = true;
    auto async_exec = graph.compile(out_spec);
    std::vector<std::pair<json::String, std::shared_ptr<json::Value>>> rst_vals;
    auto on_opr = [&m_opr_footprint, &rst_vals](cg::OperatorNodeBase* opr) {
        Result trait(m_opr_footprint.calc_footprint(opr));
        rst_vals.emplace_back(json::String(opr->id_str()), trait.to_json());
        return true;
    };
    async_exec->iter_opr_seq(on_opr);
    auto opr_fp = json::Object::make(rst_vals);
    return json::Object::make(
            {{"opr_footprint", opr_fp}, {"graph_exec", async_exec->to_json()}});
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
