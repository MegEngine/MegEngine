/**
 * \file imperative/src/impl/ops/autogen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

// FIXME: split this file into separate files for each specialized op

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/fake_quant.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/roi_align.h"
#include "megbrain/opr/dnn/roi_pooling.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/rand.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include "../op_trait.h"

namespace mgb::imperative {

namespace { namespace convolution {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Convolution>();
    return Convolution::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution&>(def);
    return opr::Convolution::make(inputs[0], inputs[1], conv.param(), conv.policy());
}

OP_TRAIT_REG(Convolution, Convolution, opr::Convolution)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // convolution

namespace { namespace convolution_backward_data {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvolutionBackwardData&>(def);
    cg::OperatorNodeConfig config;
    if (inputs.size() == 2) {
        return opr::ConvolutionBackwardData::make(inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else {
        mgb_assert(inputs.size() == 3);
        return opr::ConvolutionBackwardData::make(inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    }
}

OP_TRAIT_REG(ConvolutionBackwardData, ConvolutionBackwardData)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // convolution_backward_data

namespace { namespace dimshuffle {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Dimshuffle>();
    std::vector<int> pattern(node->param().pattern_len);
    for (size_t i = 0; i < node->param().pattern_len; ++ i) {
        pattern[i] = node->param().pattern[i];
    }
    return Dimshuffle::make(pattern);
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& ds = static_cast<const Dimshuffle&>(def);
    return opr::Dimshuffle::make(inputs[0], ds.pattern);
}

OP_TRAIT_REG(Dimshuffle, Dimshuffle, opr::Dimshuffle)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // dimshuffle

namespace { namespace add_axis {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& add_axis = static_cast<const AddAxis&>(def);
    using Desc = opr::AxisAddRemove::AxisDesc;
    std::vector<Desc> param;
    for (auto&& i : add_axis.axis) {
        param.push_back(Desc::make_add(i));
    }
    return opr::AxisAddRemove::make(inputs[0], param);
}

OP_TRAIT_REG(AddAxis, AddAxis)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // add_axis

namespace { namespace remove_axis {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& remove_axis = static_cast<const RemoveAxis&>(def);
    using Desc = opr::AxisAddRemove::AxisDesc;
    std::vector<Desc> param;
    for (auto&& i : remove_axis.axis) {
        param.push_back(Desc::make_remove(i));
    }
    return opr::AxisAddRemove::make(inputs[0], param);
}

OP_TRAIT_REG(RemoveAxis, RemoveAxis)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // remove_axis

namespace { namespace top_k {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& topk = static_cast<const TopK&>(def);
    return opr::TopK::make(inputs[0], inputs[1], topk.param())[0]
            .node()->owner_opr();
}

OP_TRAIT_REG(TopK, TopK)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // top_k

namespace { namespace reduce {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& reduce = static_cast<const Reduce&>(def);
    if (inputs.size() > 1) {
        return opr::Reduce::make(inputs[0], reduce.param(), inputs[1]);
    } else {
        return opr::Reduce::make(inputs[0], reduce.param());
    }
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Reduce>();
    return Reduce::make(node->param());
}

OP_TRAIT_REG(Reduce, Reduce, opr::Reduce)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // reduce

namespace { namespace adaptive_pooling {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& pool = static_cast<const AdaptivePooling&>(def);
    return opr::AdaptivePooling::make(inputs[0], inputs[1], pool.param());
}

OP_TRAIT_REG(AdaptivePooling, AdaptivePooling)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // adaptive_pooling

namespace { namespace conv_bias {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvBias&>(def);
    cg::OperatorNodeConfig config{conv.dtype};
    if (inputs.size() == 2) {
        return opr::ConvBias::make(inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 3) {
        return opr::ConvBias::make(inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 4) {
        return opr::ConvBias::make(inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), conv.policy(), config);
    }
    mgb_assert(0);
}

OP_TRAIT_REG(ConvBias, ConvBias)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // conv_bias

namespace { namespace batch_conv_bias {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const BatchConvBias&>(def);
    cg::OperatorNodeConfig config{conv.dtype};
    if (inputs.size() == 2) {
        return opr::BatchConvBias::make(inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 3) {
        return opr::BatchConvBias::make(inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 4) {
        return opr::BatchConvBias::make(inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), conv.policy(), config);
    }
    mgb_assert(0);
}

OP_TRAIT_REG(BatchConvBias, BatchConvBias)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // batch_conv_bias

namespace { namespace pooling {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& pool = static_cast<const Pooling&>(def);
    return opr::Pooling::make(inputs[0], pool.param());
}
OP_TRAIT_REG(Pooling, Pooling)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // pooling

namespace { namespace matrix_mul {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& matmul = static_cast<const MatrixMul&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::MatrixMul::make(inputs[0], inputs[1], matmul.param());
}
OP_TRAIT_REG(MatrixMul, MatrixMul)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // matrix_mul

namespace { namespace batched_matrix_mul {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& matmul = static_cast<const BatchedMatrixMul&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::BatchedMatrixMul::make(inputs[0], inputs[1], matmul.param());
}
OP_TRAIT_REG(BatchedMatrixMul, BatchedMatrixMul)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // batched_matrix_mul

namespace { namespace dot {
auto apply_on_var_node(
        const OpDef&,
        const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 2);
    return opr::Dot::make(inputs[0], inputs[1]);
}
OP_TRAIT_REG(Dot, Dot)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // dot

namespace { namespace argsort {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& argsort = static_cast<const Argsort&>(def);
    return opr::Argsort::make(inputs[0], argsort.param());
}
OP_TRAIT_REG(Argsort, Argsort)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // argsort

namespace { namespace argmax {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& argmax = static_cast<const Argmax&>(def);
    return opr::Argmax::make(inputs[0], argmax.param());
}
OP_TRAIT_REG(Argmax, Argmax)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // argmax

namespace { namespace argmin {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& argmin = static_cast<const Argmin&>(def);
    return opr::Argmin::make(inputs[0], argmin.param());
}
OP_TRAIT_REG(Argmin, Argmin)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // argmin

namespace { namespace warp_perspective {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& warp = static_cast<const WarpPerspective&>(def);
    if (inputs.size() == 3) {
        return opr::WarpPerspective::make(inputs[0], inputs[1], inputs[2], warp.param());
    } else {
        mgb_assert(inputs.size() == 4);
        return opr::WarpPerspective::make(inputs[0], inputs[1], inputs[2], inputs[3], warp.param());
    }
}
OP_TRAIT_REG(WarpPerspective, WarpPerspective)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // warp_perspective

namespace { namespace group_local {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& local = static_cast<const GroupLocal&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::GroupLocal::make(inputs[0], inputs[1], local.param());
}
OP_TRAIT_REG(GroupLocal, GroupLocal)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // group_local

namespace { namespace indexing_one_hot {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const IndexingOneHot&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::IndexingOneHot::make(inputs[0], inputs[1], op.param());
}
OP_TRAIT_REG(IndexingOneHot, IndexingOneHot)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // indexing_one_hot

namespace { namespace indexing_set_one_hot {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const IndexingSetOneHot&>(def);
    mgb_assert(inputs.size() == 3);
    return opr::IndexingSetOneHot::make(inputs[0], inputs[1], inputs[2], op.param());
}
OP_TRAIT_REG(IndexingSetOneHot, IndexingSetOneHot)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // indexing_set_one_hot

namespace { namespace typecvt {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const TypeCvt&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::TypeCvt::make(inputs[0], op.dtype);
}
OP_TRAIT_REG(TypeCvt, TypeCvt)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // typecvt

namespace { namespace concat {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Concat&>(def);
    cg::OperatorNodeConfig config{op.comp_node};
    return opr::Concat::make(inputs, op.axis, config);
}
OP_TRAIT_REG(Concat, Concat)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // concat

namespace { namespace copy {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Copy&>(def);
    mgb_assert(inputs.size() == 1);
    cg::OperatorNodeConfig config{op.comp_node};
    return opr::Copy::make(inputs[0], config);
}
OP_TRAIT_REG(Copy, Copy)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // copy

namespace { namespace identity {
auto apply_on_var_node(
        const OpDef&,
        const VarNodeArray& inputs) {
    mgb_assert(inputs.size() == 1);
    return opr::Identity::make(inputs[0]);
}
OP_TRAIT_REG(Identity, Identity)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // identity

namespace { namespace uniform_rng {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const UniformRNG&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::UniformRNG::make(inputs[0], op.param());
}
OP_TRAIT_REG(UniformRNG, UniformRNG)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // uniform_rng

namespace { namespace gaussian_rng {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const GaussianRNG&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::GaussianRNG::make(inputs[0], op.param());
}
OP_TRAIT_REG(GaussianRNG, GaussianRNG)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // gaussian_rng

namespace { namespace roi_align {
VarNodeArray apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIAlign&>(def);
    mgb_assert(inputs.size() == 2);
    auto* opr = opr::ROIAlign::make(inputs[0], inputs[1], op.param()).node()->owner_opr();
    return {opr->output(0), opr->output(1)};
}
OP_TRAIT_REG(ROIAlign, ROIAlign)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // roi_align

#if MGB_CUDA
namespace { namespace nvof {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const NvOf&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::NvOf::make(inputs[0], op.param());
}
OP_TRAIT_REG(NvOf, NvOf)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // nvof
#endif

namespace { namespace linspace {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Linspace&>(def);
    mgb_assert(inputs.size() == 3);
    cg::OperatorNodeConfig config{op.comp_node};
    return opr::Linspace::make(inputs[0], inputs[1], inputs[2], op.param(), config);
}
OP_TRAIT_REG(Linspace, Linspace)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // linspace

namespace { namespace eye {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Eye&>(def);
    mgb_assert(inputs.size() == 1);
    cg::OperatorNodeConfig config{op.comp_node};
    opr::Eye::Param param{op.k, op.dtype.enumv()};
    return opr::Eye::make(inputs[0], param, config);
}
OP_TRAIT_REG(Eye, Eye)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // eye

namespace { namespace roi_pooling {
VarNodeArray apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIPooling&>(def);
    mgb_assert(inputs.size() == 3);
    auto* opr = opr::ROIPooling::make(inputs[0], inputs[1], inputs[2], op.param()).node()->owner_opr();
    return {opr->output(0), opr->output(1)};
}
OP_TRAIT_REG(ROIPooling, ROIPooling)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // roi_pooling

namespace { namespace remap {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Remap&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::Remap::make(inputs[0], inputs[1], op.param());
}
OP_TRAIT_REG(Remap, Remap)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // remap

namespace { namespace reshape {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const Reshape&>(def);
    mgb_assert(inputs.size() == 2);
    return opr::Reshape::make(inputs[0], inputs[1], op.param());
}
OP_TRAIT_REG(Reshape, Reshape)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // reshape

namespace {
auto get_index(
    const VarNodeArray& inputs, size_t vidx,
    const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& mask) {
    size_t length = mask.size();
    opr::Subtensor::IndexDesc ret(length);
    for (size_t i = 0; i < length; ++ i) {
        auto&& [axis, begin, end, step, idx] = mask[i];
        ret[i].axis = axis;
        if (idx) {
            ret[i].idx = inputs[vidx++];
        } else {
            mgb_assert(begin || end || step);
            if (begin) ret[i].begin = inputs[vidx++];
            if (end) ret[i].end = inputs[vidx++];
            if (step) ret[i].step = inputs[vidx++];
        }
    }
    mgb_assert(vidx == inputs.size());
    return ret;
}
#define IN1 inputs[0]
#define IN2 inputs[0], inputs[1]

#define FANCY_INDEXING_IMPL(NAME, NR_INPUT) \
namespace NAME##_impl { \
auto apply_on_var_node( \
        const OpDef& def, \
        const VarNodeArray& inputs) { \
    auto&& op = static_cast<const NAME&>(def); \
    return opr::NAME::make(IN##NR_INPUT, get_index(inputs, NR_INPUT, op.items)); \
} \
OP_TRAIT_REG(NAME, NAME) \
    .apply_on_var_node(apply_on_var_node) \
    .fallback(); \
}

FANCY_INDEXING_IMPL(Subtensor, 1)
FANCY_INDEXING_IMPL(SetSubtensor, 2)
FANCY_INDEXING_IMPL(IncrSubtensor, 2)
FANCY_INDEXING_IMPL(IndexingMultiAxisVec, 1)
FANCY_INDEXING_IMPL(IndexingSetMultiAxisVec, 2)
FANCY_INDEXING_IMPL(IndexingIncrMultiAxisVec, 2)
FANCY_INDEXING_IMPL(MeshIndexing, 1)
FANCY_INDEXING_IMPL(IncrMeshIndexing, 2)
FANCY_INDEXING_IMPL(SetMeshIndexing, 2)
FANCY_INDEXING_IMPL(BatchedMeshIndexing, 1)
FANCY_INDEXING_IMPL(BatchedIncrMeshIndexing, 2)
FANCY_INDEXING_IMPL(BatchedSetMeshIndexing, 2)

#undef FANCY_INDEXING_IMPL
#undef IN1
#undef IN2
} // anonymous namespace

namespace { namespace fake_quant {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const FakeQuant&>(def);
    mgb_assert(inputs.size() == 3);
    return opr::FakeQuant::make(inputs[0], inputs[1], inputs[2], op.param());
}
OP_TRAIT_REG(FakeQuant, FakeQuant)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // fake_quant
namespace { namespace elemwise_multi_type {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const ElemwiseMultiType&>(def);
    OperatorNodeConfig config{op.dtype};
    return opr::ElemwiseMultiType::make(inputs, op.param(), config);
}
OP_TRAIT_REG(ElemwiseMultiType, ElemwiseMultiType)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // fake_quant

namespace { namespace svd {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = static_cast<const SVD&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::SVD::make(inputs[0], op.param())[0].node()->owner_opr()->usable_output();
}
OP_TRAIT_REG(SVD, SVD)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // svd

} // namespace mgb::imperative
