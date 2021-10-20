/**
 * \file src/gopt/impl/opr_format_modifier.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_format_modifier.h"
#include "./utils.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/sereg.h"

#include "midout.h"
MIDOUT_DECL(megbrain_opr_format_modifier)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_format_modifier, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace opr;

namespace {
template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller2 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 2) {
            return Opr::make(inputs[0], inputs[1], param, execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller3 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 3) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], param, execution_policy,
                           config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller4 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 4) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], inputs[3], param,
                           execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller5 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 5) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], param,
                           execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCallerEmpty {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray&, const typename MegDNNConv::Param&,
            const megdnn::param::ExecutionPolicy&, const OperatorNodeConfig&) {
        return nullptr;
    }
};

template <
        class Opr, class Maker0, class MegDNNConv,
        class Maker1 = MakeConvCallerEmpty<MegDNNConv>,
        class Maker2 = MakeConvCallerEmpty<MegDNNConv>,
        typename ConvParam = megdnn::param::Convolution>
struct ConvMakerImpl {
    static VarNode* make(
            const cg::VarNodeArray& inputs, const ConvParam& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        VarNode* ret =
                Maker0::template make<Opr>(inputs, param, execution_policy, config);
        if (!ret) {
            ret = Maker1::template make<Opr>(inputs, param, execution_policy, config);
        }
        if (!ret) {
            ret = Maker2::template make<Opr>(inputs, param, execution_policy, config);
        }
        mgb_assert(ret);
        return ret;
    }
};

template <typename Opr>
struct ConvMaker;

template <>
struct ConvMaker<opr::Convolution>
        : public ConvMakerImpl<
                  opr::Convolution, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution> {};
template <>
struct ConvMaker<opr::ConvolutionBackwardData>
        : public ConvMakerImpl<
                  opr::ConvolutionBackwardData, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution, MakeConvCaller3<megdnn::Convolution>> {};

template <>
struct ConvMaker<opr::ConvBiasForward>
        : public ConvMakerImpl<
                  opr::ConvBiasForward, MakeConvCaller2<megdnn::ConvBiasForward>,
                  megdnn::ConvBiasForward, MakeConvCaller3<megdnn::ConvBiasForward>,
                  MakeConvCaller4<megdnn::ConvBiasForward>, megdnn::param::ConvBias> {};
template <>
struct ConvMaker<opr::BatchConvBiasForward>
        : public ConvMakerImpl<
                  opr::BatchConvBiasForward,
                  MakeConvCaller2<megdnn::BatchConvBiasForward>,
                  megdnn::BatchConvBiasForward,
                  MakeConvCaller3<megdnn::BatchConvBiasForward>,
                  MakeConvCaller4<megdnn::BatchConvBiasForward>,
                  megdnn::param::BatchConvBias> {};

#include "../../opr/impl/internal/invoke.h"
template <typename Opr>
struct MultiAlgoOprTrait;

#define APPLY(statement, ...)                               \
    mgb::apply(                                             \
            [&](const auto&... args) { return statement; }, \
            std::tuple_cat(__VA_ARGS__))

#define INST(_Opr)                                                                     \
    template <>                                                                        \
    struct MultiAlgoOprTrait<_Opr> {                                                   \
        static constexpr bool has_algo = true;                                         \
        using MegDNNOpr = megdnn::_Opr;                                                \
        static constexpr int arity = OprArityTrait<MegDNNOpr>::arity;                  \
        using FixedTensorLayouts = std::array<TensorLayout, arity>;                    \
        static bool has_available_algo(                                                \
                const VarNodeArray& i, const cg::OperatorNodeBase* opr_) {             \
            MIDOUT_B(                                                                  \
                    midout_iv(MGB_HASH_STR(#_Opr)),                                    \
                    midout_iv(MGB_HASH_STR("has_available_algo")))                     \
            auto&& opr = opr_->cast_final_safe<_Opr>();                                \
            auto&& megdnn_opr = reinterpret_cast<MegDNNOpr*>(opr.megdnn_opr());        \
            FixedTensorLayouts array_layouts;                                          \
            size_t in = i.size() - 1;                                                  \
            for (size_t idx = 0; idx < in; idx++) {                                    \
                const auto& v = i[idx];                                                \
                array_layouts[idx] =                                                   \
                        TensorLayout{v->shape(), v->dtype(), v->format()};             \
            }                                                                          \
            const auto& v = i[in];                                                     \
            array_layouts[arity - 1] =                                                 \
                    TensorLayout{v->shape(), v->dtype(), v->format()};                 \
            return APPLY(                                                              \
                    ::megdnn::has_available_algo(megdnn_opr, args...), array_layouts); \
            MIDOUT_E                                                                   \
        }                                                                              \
    };
INST(Convolution)
INST(ConvBiasForward)
INST(ConvolutionBackwardData)
INST(PoolingForward)
#undef APPLY
#undef INST

VarNode* modify_concat_opr_format(
        gopt::intl::OprFormatInfo::TensorFormatsInfo tensor_formats,
        const VarNodeArray& i, const cg::OperatorNodeBase* opr) {
    auto base_format = tensor_formats.from;
    auto tensor_format = tensor_formats.to;
    int axis = opr->cast_final_safe<Concat>().axis();
    /// modify axis
    using Dimension = megdnn::Dimension;
    static constexpr uint32_t UNDETERMINED_EXTENT = Dimension::UNDETERMINED_EXTENT;
    auto orig_shape = tensor_formats_to_named_tensor_shape(base_format);
    auto target_shape = tensor_formats_to_named_tensor_shape(tensor_format);
    mgb_assert(
            static_cast<size_t>(axis) < orig_shape.ndim,
            "invalid axis of concat opr(axis:%d,shp:%s)", axis,
            orig_shape.to_string().c_str());
    if (orig_shape[axis].extent() != UNDETERMINED_EXTENT)
        return nullptr;
    auto axis_name = orig_shape[axis].name();
    int new_axis = target_shape.ndim;
    for (size_t i = 0; i < target_shape.ndim; ++i) {
        if (target_shape[i].name() == axis_name &&
            target_shape[i].extent() == UNDETERMINED_EXTENT) {
            new_axis = i;
            break;
        }
    }
    if (static_cast<size_t>(new_axis) >= target_shape.ndim)
        return nullptr;
    return opr::Concat::make(i, new_axis, opr->config()).node();
}
}  // namespace

namespace mgb {
namespace gopt {
namespace intl {

template <typename Opr>
struct OprFormatModifier;

#define INST(_Opr)                                                   \
    template <>                                                      \
    struct OprFormatModifier<_Opr> {                                 \
        using OprFormat = typename _Opr::Param::Format;              \
        static VarNode* make(                                        \
                OprFormat opr_format, const VarNodeArray& i,         \
                const cg::OperatorNodeBase* opr_) {                  \
            MIDOUT_B(_Opr)                                           \
            auto&& opr = opr_->cast_final_safe<_Opr>();              \
            auto param = opr.param();                                \
            param.format = opr_format;                               \
            return ConvMaker<_Opr>::make(                            \
                    i, param, opr.execution_policy(), opr.config()); \
            MIDOUT_E                                                 \
        }                                                            \
    };
INST(Convolution);
INST(ConvBiasForward);
INST(ConvolutionBackwardData);
INST(BatchConvBiasForward);
#undef INST

template <>
struct OprFormatModifier<WarpPerspective> {
    using Opr = opr::WarpPerspective;
    using OprFormat = typename Opr::Param::Format;
    static VarNode* make(
            OprFormat opr_format, const VarNodeArray& i,
            const cg::OperatorNodeBase* opr_) {
        MIDOUT_B(Opr)
        auto&& opr = opr_->cast_final_safe<Opr>();
        auto param = opr.param();
        param.format = opr_format;
        if (i.size() == 3) {
            return Opr::make(i[0], i[1], i[2], param, opr.config()).node();
        } else {
            mgb_assert(i.size() == 4);
            return Opr::make(i[0], i[1], i[2], i[3], param, opr.config()).node();
        }
        MIDOUT_E
    }
};

#define INST(_Opr, _arity)                                               \
    template <>                                                          \
    struct OprFormatModifier<_Opr> {                                     \
        using OprFormat = typename _Opr::Param::Format;                  \
        static VarNode* make(                                            \
                OprFormat opr_format, const VarNodeArray& i,             \
                const cg::OperatorNodeBase* opr_) {                      \
            MIDOUT_B(_Opr)                                               \
            auto&& opr = opr_->cast_final_safe<_Opr>();                  \
            auto param = opr.param();                                    \
            param.format = opr_format;                                   \
            return serialization::OprMaker<_Opr, _arity>::make(          \
                           param, i, *i[0]->owner_graph(), opr.config()) \
                    ->output(0);                                         \
            MIDOUT_E                                                     \
        }                                                                \
    };
INST(PoolingForward, 1);
INST(Resize, 2);
#undef INST

VarNode* modify_opr_format(
        OprFormatInfo opr_format_info, const VarNodeArray& i,
        const cg::OperatorNodeBase* opr) {
#define cb(_Opr)                                                                  \
    if (opr->dyn_typeinfo() == _Opr::typeinfo()) {                                \
        return OprFormatModifier<_Opr>::make(opr_format_info.opr_format, i, opr); \
    } else
    FOREACH_FORMAT_AWARE_OPR(cb)
    if (opr->dyn_typeinfo() == opr::Concat::typeinfo()) {
        return modify_concat_opr_format(opr_format_info.tensor_formats, i, opr);
    } else {
        mgb_throw(
                InternalError, "invalid format aware operator(got:%s)",
                opr->dyn_typeinfo()->name);
    }
#undef cb
}

bool has_available_algo(const VarNodeArray& i, const cg::OperatorNodeBase* opr) {
#define cb(_Opr)                                                    \
    if (opr->dyn_typeinfo() == _Opr::typeinfo()) {                  \
        MGB_MARK_USED_VAR(MultiAlgoOprTrait<_Opr>::has_algo);       \
        VarNodeArray _ = i;                                         \
        _.emplace_back(opr->output(0));                             \
        return MultiAlgoOprTrait<_Opr>::has_available_algo(_, opr); \
    } else
    cb(Convolution) cb(ConvBiasForward) cb(ConvolutionBackwardData) cb(PoolingForward) {
        mgb_throw(
                InternalError, "invalid multi-algo operator(got:%s)",
                opr->dyn_typeinfo()->name);
    }
#undef cb
}

bool has_opr_format(const cg::OperatorNodeBase* opr) {
    bool ret = false;
#define cb(_Opr) ret |= opr->dyn_typeinfo() == _Opr::typeinfo();
    FOREACH_FORMAT_AWARE_OPR(cb)
#undef cb
    return ret;
}

bool has_opr_format_modifier(const cg::OperatorNodeBase* opr) {
    bool ret = false;
#define cb(_Opr) ret |= opr->dyn_typeinfo() == _Opr::typeinfo();
    FOREACH_MODIFY_OPR_FORMAT_OPR(cb)
#undef cb
    return ret;
}

bool allow_aligned_layout(const cg::OperatorNodeBase* opr) {
    return opr->dyn_typeinfo() != opr::Concat::typeinfo() &&
           opr->dyn_typeinfo() != opr::Reduce::typeinfo();
}

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
