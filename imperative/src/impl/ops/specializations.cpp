/**
 * \file imperative/src/impl/ops/specialzations.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

// FIXME: split this file into separate files for each specialized op

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/correlation.h"
#include "megbrain/opr/dnn/fake_quant.h"
#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/opr/dnn/layer_norm.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/opr/dnn/lsq.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/dnn/roi_align.h"
#include "megbrain/opr/dnn/roi_pooling.h"
#include "megbrain/opr/dnn/sliding_window_transpose.h"
#include "megbrain/opr/dnn/tqt.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/misc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/rand.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include "../blob_manager_impl.h"
#include "../op_trait.h"

namespace mgb::imperative {

namespace {
namespace dimshuffle {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Dimshuffle>();
    std::vector<int> pattern(node->param().pattern_len);
    for (size_t i = 0; i < node->param().pattern_len; ++i) {
        pattern[i] = node->param().pattern[i];
    }
    return Dimshuffle::make(pattern);
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& ds = static_cast<const Dimshuffle&>(def);
    OperatorNodeConfig config{ds.make_name()};
    return opr::Dimshuffle::make(inputs[0], ds.pattern, 0UL, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& ds = static_cast<const Dimshuffle&>(def);
    mgb_assert(
            ds.pattern.size() <= TensorShape::MAX_NDIM,
            "Dimshuffle pattern exceeds max length of %zd", TensorShape::MAX_NDIM);
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 1, "Dimshuffle expects 1 inputs; got %lu actually", nr_inp);
    auto&& src = inputs[0];
    auto inp_layout = src->layout();
    size_t pattern_ndim = *std::max_element(ds.pattern.begin(), ds.pattern.end()) + 1;
    mgb_assert(
            inp_layout.ndim == pattern_ndim,
            "input ndim mismatch for Dimshuffle: expect=%zd actual=%zd", pattern_ndim,
            inp_layout.ndim);
    TensorLayout out_layout{inp_layout.dtype};
    out_layout.ndim = ds.pattern.size();

    size_t idx = 0;
    bool input_used[TensorLayout::MAX_NDIM] = {0};
    for (auto i : ds.pattern) {
        if (i < 0) {
            out_layout.shape[idx] = 1;
            out_layout.stride[idx] = 1;
        } else {
            input_used[i] = true;
            out_layout.shape[idx] = inp_layout.shape[i];
            out_layout.stride[idx] = inp_layout.stride[i];
        }
        ++idx;
    }
    if (out_layout.is_contiguous()) {
        out_layout.init_contiguous_stride();
    }
    for (size_t i = 0; i < pattern_ndim; ++i) {
        mgb_assert(
                input_used[i] || inp_layout.shape[i] == 1,
                "non-1 dim discarded in Dimshuffle: ishp=%s dim=%zd",
                inp_layout.megdnn::TensorShape::to_string().c_str(), i);
    }
    // memory forward
    return {Tensor::make(src->blob(), src->offset(), out_layout)};
}

OP_TRAIT_REG(Dimshuffle, Dimshuffle, opr::Dimshuffle)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace dimshuffle
}  // namespace

namespace {
namespace add_axis {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& add_axis = static_cast<const AddAxis&>(def);
    using Desc = opr::AxisAddRemove::AxisDesc;
    std::vector<Desc> param;
    for (auto&& i : add_axis.axis) {
        param.push_back(Desc::make_add(i));
    }
    OperatorNodeConfig config{add_axis.make_name()};
    return opr::AxisAddRemove::make(inputs[0], param, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<AddAxis>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 1, "AddAxis expects 1 inputs; got %lu actually", nr_inp);
    auto&& src = inputs[0];
    auto tlayout = src->layout();
    for (auto&& i : op_def.axis) {
        tlayout.add_axis_cont_inplace(i);
    }
    // memory forward
    return {Tensor::make(src->blob(), src->offset(), tlayout)};
}

OP_TRAIT_REG(AddAxis, AddAxis)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace add_axis
}  // namespace

namespace {
namespace remove_axis {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& remove_axis = static_cast<const RemoveAxis&>(def);
    using Desc = opr::AxisAddRemove::AxisDesc;
    std::vector<Desc> param;
    for (auto&& i : remove_axis.axis) {
        param.push_back(Desc::make_remove(i));
    }
    OperatorNodeConfig config{remove_axis.make_name()};
    return opr::AxisAddRemove::make(inputs[0], param, config);
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<RemoveAxis>();
    size_t nr_inp = inputs.size();
    mgb_assert(nr_inp == 1, "RemoveAxis expects 1 inputs; got %lu actually", nr_inp);
    auto&& src = inputs[0];
    auto tlayout = src->layout();
    for (auto&& i : op_def.axis) {
        if (tlayout.ndim == 1) {
            mgb_assert(
                    tlayout.shape[0] == 1 && i == 0,
                    "can not remove axis %u from tensor of shape=%s", i,
                    tlayout.megdnn::TensorShape::to_string().c_str());
        } else {
            mgb_assert(
                    i < tlayout.ndim && tlayout.shape[i] == 1,
                    "can not remove axis %u from tensor of shape=%s", i,
                    tlayout.megdnn::TensorShape::to_string().c_str());
            tlayout.remove_axis_inplace(i);
        }
    }
    // memory forward
    return {Tensor::make(src->blob(), src->offset(), tlayout)};
}

OP_TRAIT_REG(RemoveAxis, RemoveAxis)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace remove_axis
}  // namespace

namespace {
namespace top_k {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& topk = static_cast<const TopK&>(def);
    OperatorNodeConfig config{topk.make_name()};
    return opr::TopK::make(inputs[0], inputs[1], topk.param(), config)[0]
            .node()
            ->owner_opr();
}

OP_TRAIT_REG(TopK, TopK).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace top_k
}  // namespace

namespace {
namespace adaptive_pooling {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& pool = static_cast<const AdaptivePooling&>(def);
    OperatorNodeConfig config{pool.make_name()};
    return opr::AdaptivePooling::make(inputs[0], inputs[1], pool.param(), config);
}

OP_TRAIT_REG(AdaptivePooling, AdaptivePooling)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace adaptive_pooling
}  // namespace

namespace {
namespace conv_bias {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvBias&>(def);
    cg::OperatorNodeConfig config{conv.dtype};
    config.name(conv.make_name());
    if (inputs.size() == 2) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 3) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 4) {
        return opr::ConvBias::make(
                inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), conv.policy(),
                config);
    }
    mgb_assert(0);
}

OP_TRAIT_REG(ConvBias, ConvBias).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace conv_bias
}  // namespace

namespace {
namespace batch_conv_bias {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& conv = static_cast<const BatchConvBias&>(def);
    cg::OperatorNodeConfig config{conv.dtype};
    config.name(conv.make_name());
    if (inputs.size() == 2) {
        return opr::BatchConvBias::make(
                inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 3) {
        return opr::BatchConvBias::make(
                inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    } else if (inputs.size() == 4) {
        return opr::BatchConvBias::make(
                inputs[0], inputs[1], inputs[2], inputs[3], conv.param(), conv.policy(),
                config);
    }
    mgb_assert(0);
}

OP_TRAIT_REG(BatchConvBias, BatchConvBias)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace batch_conv_bias
}  // namespace

namespace {
namespace pooling {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& pool = static_cast<const Pooling&>(def);
    OperatorNodeConfig config{pool.make_name()};
    return opr::Pooling::make(inputs[0], pool.param(), pool.policy(), config);
}
OP_TRAIT_REG(Pooling, Pooling).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace pooling
}  // namespace

namespace {
namespace matrix_mul {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& matmul = static_cast<const MatrixMul&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{matmul.make_name()};
    return opr::MatrixMul::make(
            inputs[0], inputs[1], matmul.param(), matmul.policy(), config);
}
OP_TRAIT_REG(MatrixMul, MatrixMul).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace matrix_mul
}  // namespace

namespace {
namespace batched_matrix_mul {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& matmul = static_cast<const BatchedMatrixMul&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{matmul.make_name()};
    return opr::BatchedMatrixMul::make(
            inputs[0], inputs[1], matmul.param(), matmul.policy(), config);
}
OP_TRAIT_REG(BatchedMatrixMul, BatchedMatrixMul)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace batched_matrix_mul
}  // namespace

namespace {
namespace dot {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Dot>();
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Dot::make(inputs[0], inputs[1], config);
}

// std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
//     auto* node = &node_->cast_final_safe<opr::Dot>();
//     return Dot::make(node->param());
// }

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto a = inputs[0]->layout();
    auto comp_node = inputs[0]->comp_node();
    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds;
    inp_tensornds.reserve(inputs.size());
    auto dnn_opr = opr::intl::create_megdnn_opr<megdnn::Dot>(comp_node);
    for (unsigned i = 0; i < inputs.size(); ++i) {
        auto dnn_ten = inputs[i]->dnn_tensor();
        inp_tensornds.push_back(dnn_ten);
    }
    TensorLayout oup_layout{inputs[0]->dtype()};
    auto inp1_tensor = inputs[0]->dnn_tensor();
    auto inp2_tensor = inputs[1]->dnn_tensor();
    dnn_opr->deduce_layout(inp1_tensor.layout, inp2_tensor.layout, oup_layout);

    if (inputs[0]->layout().is_empty() || inputs[1]->layout().is_empty()) {
        auto fill_opr = opr::intl::create_megdnn_opr<megdnn::Fill>(comp_node);
        DeviceTensorND out =
                BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);
        fill_opr->param() = 0;
        fill_opr->exec(out.as_megdnn(), {});
        return {Tensor::make(out)};
    }

    auto wk_size = dnn_opr->get_workspace_in_bytes(
            inp_tensornds[0].layout, inp_tensornds[1].layout, output_descs[0].layout);

    DeviceTensorND out_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);
    TensorLayout wk_layout{TensorShape{wk_size}, inputs[0]->dtype()};
    DeviceTensorND workspace =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, wk_layout);
    megdnn::Workspace dnn_wk(workspace.raw_ptr(), wk_size);

    dnn_opr->exec(
            inp_tensornds[0], inp_tensornds[1], out_devtensor.as_megdnn(), dnn_wk);

    return {Tensor::make(out_devtensor)};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<Dot>();
    SmallVector<LogicalTensorDesc> dests(1);
    dests[0].layout = TensorLayout(TensorShape{1}, inputs[0].layout.dtype);
    dests[0].comp_node = inputs[0].comp_node;
    return {dests, true};
}

OP_TRAIT_REG(Dot, Dot, opr::Dot)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace dot
}  // namespace

namespace {
namespace argsort {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& argsort = static_cast<const Argsort&>(def);
    OperatorNodeConfig config{argsort.make_name()};
    return opr::Argsort::make(inputs[0], argsort.param(), config);
}
OP_TRAIT_REG(Argsort, Argsort).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace argsort
}  // namespace

namespace {
namespace argmax {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& argmax = static_cast<const Argmax&>(def);
    OperatorNodeConfig config{argmax.make_name()};
    return opr::Argmax::make(inputs[0], argmax.param(), config);
}
OP_TRAIT_REG(Argmax, Argmax).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace argmax
}  // namespace

namespace {
namespace argmin {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& argmin = static_cast<const Argmin&>(def);
    OperatorNodeConfig config{argmin.make_name()};
    return opr::Argmin::make(inputs[0], argmin.param(), config);
}
OP_TRAIT_REG(Argmin, Argmin).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace argmin
}  // namespace

namespace {
namespace warp_perspective {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& warp = static_cast<const WarpPerspective&>(def);
    OperatorNodeConfig config{warp.make_name()};
    if (inputs.size() == 3) {
        return opr::WarpPerspective::make(
                inputs[0], inputs[1], inputs[2], warp.param(), config);
    } else {
        mgb_assert(inputs.size() == 4);
        return opr::WarpPerspective::make(
                inputs[0], inputs[1], inputs[2], inputs[3], warp.param(), config);
    }
}
OP_TRAIT_REG(WarpPerspective, WarpPerspective)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace warp_perspective
}  // namespace

namespace {
namespace group_local {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& local = static_cast<const GroupLocal&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{local.make_name()};
    return opr::GroupLocal::make(inputs[0], inputs[1], local.param(), config);
}
OP_TRAIT_REG(GroupLocal, GroupLocal).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace group_local
}  // namespace

namespace {
namespace indexing_set_one_hot {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const IndexingSetOneHot&>(def);
    mgb_assert(inputs.size() == 3);
    OperatorNodeConfig config{op.make_name()};
    return opr::IndexingSetOneHot::make(
            inputs[0], inputs[1], inputs[2], op.param(), config);
}
OP_TRAIT_REG(IndexingSetOneHot, IndexingSetOneHot)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace indexing_set_one_hot
}  // namespace

namespace {
namespace typecvt {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const TypeCvt&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::TypeCvt::make(inputs[0], op.dtype, config);
}
OP_TRAIT_REG(TypeCvt, TypeCvt).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace typecvt
}  // namespace

namespace {
namespace concat {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Concat&>(def);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());
    return opr::Concat::make(inputs, op.axis, config);
}
OP_TRAIT_REG(Concat, Concat).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace concat
}  // namespace

namespace {
namespace copy {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Copy&>(def);
    mgb_assert(inputs.size() == 1);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());
    return opr::Copy::make(inputs[0], config);
}
OP_TRAIT_REG(Copy, Copy).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace copy
}  // namespace

namespace {
namespace assert_equal {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final<AssertEqual>();
    if (inputs.size() == 2) {
        return opr::AssertEqual::make(inputs[0], inputs[1], op.param());
    } else {
        // workaround for MiniGraph, which only allow one opr in the graph
        mgb_assert(inputs.size() == 3);
        return opr::AssertEqual::make(inputs[0], inputs[1], inputs[2], op.param(), {});
    }
}

OP_TRAIT_REG(AssertEqual, AssertEqual).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace assert_equal
}  // namespace

namespace {
namespace roi_align {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIAlign&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    auto* opr = opr::ROIAlign::make(inputs[0], inputs[1], op.param(), config)
                        .node()
                        ->owner_opr();
    return {opr->output(0), opr->output(1)};
}
OP_TRAIT_REG(ROIAlign, ROIAlign).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace roi_align
}  // namespace

namespace {
namespace correlation {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Correlation&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Correlation::make(inputs[0], inputs[1], op.param(), config);
}
OP_TRAIT_REG(Correlation, Correlation).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace correlation
}  // namespace

#if MGB_CUDA
namespace {
namespace nvof {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const NvOf&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::NvOf::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(NvOf, NvOf).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace nvof
}  // namespace
#endif

namespace {
namespace linspace {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Linspace&>(def);
    mgb_assert(inputs.size() == 3);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());
    return opr::Linspace::make(inputs[0], inputs[1], inputs[2], op.param(), config);
}
OP_TRAIT_REG(Linspace, Linspace).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace linspace
}  // namespace

namespace {
namespace eye {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Eye&>(def);
    mgb_assert(inputs.size() == 1);
    cg::OperatorNodeConfig config{op.comp_node};
    config.name(op.make_name());
    opr::Eye::Param param{op.k, op.dtype.enumv()};
    return opr::Eye::make(inputs[0], param, config);
}
OP_TRAIT_REG(Eye, Eye).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace eye
}  // namespace

namespace {
namespace diag {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Diag&>(def);
    mgb_assert(inputs.size() == 1);
    cg::OperatorNodeConfig config{op.make_name()};
    opr::Diag::Param param{op.k};
    return opr::Diag::make(inputs[0], param, config);
}
OP_TRAIT_REG(Diag, Diag).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace diag
}  // namespace

namespace {
namespace roi_pooling {
VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ROIPooling&>(def);
    mgb_assert(inputs.size() == 3);
    OperatorNodeConfig config{op.make_name()};
    auto* opr =
            opr::ROIPooling::make(inputs[0], inputs[1], inputs[2], op.param(), config)
                    .node()
                    ->owner_opr();
    return {opr->output(0), opr->output(1)};
}
OP_TRAIT_REG(ROIPooling, ROIPooling).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace roi_pooling
}  // namespace

namespace {
namespace remap {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Remap&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::Remap::make(inputs[0], inputs[1], op.param(), config);
}
OP_TRAIT_REG(Remap, Remap).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace remap
}  // namespace

namespace {
auto get_index(
        const VarNodeArray& inputs, size_t vidx,
        const std::vector<std::tuple<int8_t, bool, bool, bool, bool>>& mask) {
    size_t length = mask.size();
    opr::Subtensor::IndexDesc ret(length);
    for (size_t i = 0; i < length; ++i) {
        auto&& [axis, begin, end, step, idx] = mask[i];
        ret[i].axis = axis;
        if (idx) {
            ret[i].idx = inputs[vidx++];
        } else {
            mgb_assert(begin || end || step);
            if (begin)
                ret[i].begin = inputs[vidx++];
            if (end)
                ret[i].end = inputs[vidx++];
            if (step)
                ret[i].step = inputs[vidx++];
        }
    }
    mgb_assert(vidx == inputs.size());
    return ret;
}
#define IN1 inputs[0]
#define IN2 inputs[0], inputs[1]

#define FANCY_INDEXING_IMPL(NAME, NR_INPUT)                                       \
    namespace NAME##_impl {                                                       \
        auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {    \
            auto&& op = static_cast<const NAME&>(def);                            \
            OperatorNodeConfig config{op.make_name()};                            \
            return opr::NAME::make(                                               \
                    IN##NR_INPUT, get_index(inputs, NR_INPUT, op.items), config); \
        }                                                                         \
        OP_TRAIT_REG(NAME, NAME).apply_on_var_node(apply_on_var_node).fallback(); \
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
}  // anonymous namespace

namespace {
namespace fake_quant {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const FakeQuant&>(def);
    mgb_assert(inputs.size() == 3);
    OperatorNodeConfig config{op.make_name()};
    return opr::FakeQuant::make(inputs[0], inputs[1], inputs[2], op.param(), config);
}
OP_TRAIT_REG(FakeQuant, FakeQuant).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace fake_quant
}  // namespace

namespace {
namespace tqt {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const TQT&>(def);
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{op.make_name()};
    return opr::TQT::make(inputs[0], inputs[1], op.param(), config);
}
OP_TRAIT_REG(TQT, TQT).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace tqt
}  // namespace

namespace {
namespace elemwise_multi_type {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const ElemwiseMultiType&>(def);
    OperatorNodeConfig config{op.dtype};
    config.name(op.make_name());
    return opr::ElemwiseMultiType::make(inputs, op.param(), config);
}
OP_TRAIT_REG(ElemwiseMultiType, ElemwiseMultiType)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace elemwise_multi_type
}  // namespace

namespace {
namespace svd {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const SVD&>(def);
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::SVD::make(inputs[0], op.param(), config)[0]
            .node()
            ->owner_opr()
            ->usable_output();
}
OP_TRAIT_REG(SVD, SVD).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace svd
}  // namespace

namespace {
namespace images2neibs {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Images2Neibs&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::Images2Neibs::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(Images2Neibs, Images2Neibs)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace images2neibs
}  // namespace

namespace {
namespace lsq {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LSQ&>(def);
    mgb_assert(inputs.size() == 4);
    OperatorNodeConfig config{op.make_name()};
    return opr::LSQ::make(
            inputs[0], inputs[1], inputs[2], inputs[3], op.param(), config);
}
OP_TRAIT_REG(LSQ, LSQ).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace lsq
}  // namespace

namespace {
namespace sliding_window_transpose {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const SlidingWindowTranspose&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::SlidingWindowTranspose::make(inputs[0], op.param(), config);
}
OP_TRAIT_REG(SlidingWindowTranspose, SlidingWindowTranspose)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // namespace sliding_window_transpose
}  // namespace

namespace {
namespace cumsum {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Cumsum&>(def);
    OperatorNodeConfig config{op.make_name()};
    return opr::Cumsum::make(inputs[0], op.param(), config);
}

OP_TRAIT_REG(Cumsum, Cumsum).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace cumsum
}  // namespace

namespace padding {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const Padding&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::Padding::make(inputs[0], op.param());
}
OP_TRAIT_REG(Padding, Padding).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace padding

namespace lrn {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LRN&>(def);
    mgb_assert(inputs.size() == 1);
    return opr::LRN::make(inputs[0], op.param());
}
OP_TRAIT_REG(LRN, LRN).apply_on_var_node(apply_on_var_node).fallback();
}  // namespace lrn

namespace layer_norm {

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = static_cast<const LayerNorm&>(def);
    size_t nr_inp = inputs.size();
    auto p = op.param();
    mgb_assert((nr_inp == 3 && p.affine) || (nr_inp == 1 && !p.affine));
    OperatorNodeConfig config{op.make_name()};
    if (nr_inp == 3) {
        return opr::LayerNorm::make(
                       inputs[0], inputs[1], inputs[2], op.param(), config)[0]
                .node()
                ->owner_opr();
    } else {
        return opr::LayerNorm::make(inputs[0], op.param(), config)[0]
                .node()
                ->owner_opr();
    }
}

OP_TRAIT_REG(LayerNorm, LayerNorm).apply_on_var_node(apply_on_var_node).fallback();

}  // namespace layer_norm

}  // namespace mgb::imperative
