#include <numeric>
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/graph/symbol_var.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"

#include "../algo_chooser.h"

namespace mgb {
namespace imperative {

namespace {
namespace matrix_mul {

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& matmul = def.cast_final_safe<MatrixMul>();
    mgb_assert(inputs.size() == 2);
    auto inp1 = SymbolVar{inputs[0]}, inp2 = SymbolVar{inputs[1]};
    auto dim1 = matmul.dimA, dim2 = matmul.dimB;

    auto cn = inputs[0]->comp_node();
    using IndexDesc = opr::Subtensor::IndexDesc;
    OperatorNodeConfig config{matmul.make_name(), cn};

    DTypeScalar vi{-1};
    auto graph = inputs[0]->owner_graph();

    SymbolVar shp1_head, shp1_tail, shp2_head, shp2_tail;
    if (dim1 > 2) {
        auto idx = opr::ImmutableTensor::make(*graph, vi, config);
        auto shp1 = inp1.symshape();
        IndexDesc head_desc(1);
        head_desc[0].end = idx;
        shp1_head = opr::Subtensor::make(shp1, head_desc);
        auto batch = opr::Reduce::make(shp1_head, {Reduce::Mode::PRODUCT, 0});
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        shp1_tail = opr::Subtensor::make(shp1, tail_desc);
        auto tshp = opr::Concat::make({batch, shp1_tail}, 0, cn);
        inp1 = inp1.reshape(tshp);
    }
    if (dim2 > 2) {
        auto idx = opr::ImmutableTensor::make(*graph, vi, config);
        auto shp2 = inp2.symshape();
        IndexDesc head_desc(1);
        head_desc[0].end = idx;
        shp2_head = opr::Subtensor::make(shp2, head_desc);
        auto batch = opr::Reduce::make(shp2_head, {Reduce::Mode::PRODUCT, 0});
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        auto shp2_tail = opr::Subtensor::make(shp2, tail_desc);
        auto tshp = opr::Concat::make({batch, shp2_tail}, 0, cn);
        inp2 = inp2.reshape(tshp);
    }
    auto result =
            opr::MatrixMul::make(inp1, inp2, matmul.param(), matmul.policy(), config);
    if (dim1 > 2) {
        auto idx = opr::ImmutableTensor::make(*graph, vi, config);
        auto result_shape = result.symshape();
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        auto shp_tail = opr::Subtensor::make(result_shape, tail_desc);
        auto tshp = opr::Concat::make({shp1_head, shp_tail}, 0, cn);
        result = result.reshape(tshp);
    }
    if (dim2 > 2) {
        auto idx = opr::ImmutableTensor::make(*graph, vi, config);
        auto result_shape = result.symshape();
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        auto shp_tail = opr::Subtensor::make(result_shape, tail_desc);
        auto tshp = opr::Concat::make({shp2_head, shp_tail}, 0, cn);
        result = result.reshape(tshp);
    }

    return result;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& matmul = def.cast_final_safe<MatrixMul>();
    auto layout1 = inputs[0].layout;
    auto layout2 = inputs[1].layout;
    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;

    DType dst_dtype;
    if (dim1 == dim2 && dim2 >= 3) {  // only happens in backward
        for (size_t i = 1; i + 1 < layout1.ndim; ++i) {
            layout1[0] *= layout1[i];
            layout2[0] *= layout2[i];
        }
        layout1[1] = layout1[layout1.ndim - 1];
        layout1.ndim = 2;
        layout1.init_contiguous_stride();
        layout2[1] = layout2[layout2.ndim - 1];
        layout2.ndim = 2;
        layout2.init_contiguous_stride();
        dim1 = dim2 = 2;
    }

    DnnOprHelper<megdnn::MatrixMul> dnn_opr(matmul.param());
    dnn_opr.opr().deduce_dtype(layout1.dtype, layout1.dtype, dst_dtype);

    if (dim1 == 0 || dim2 == 0) {
        return {{{TensorLayout(dst_dtype), inputs[0].comp_node}}, false};
    }

    if (matmul.transposeA)
        std::swap(layout1[0], layout1[1]);
    if (matmul.transposeB)
        std::swap(layout2[0], layout2[1]);

    mgb_assert(layout1[dim1 - 1] == layout2[0]);

    TensorLayout dst_layout(dst_dtype);
    size_t ci = 0;
    for (size_t i = 0; i < dim1 - 1; i++)
        dst_layout[ci++] = layout1[i];
    if (dim2 == 2)
        dst_layout[ci++] = layout2[1];
    dst_layout.ndim = ci;
    dst_layout.init_contiguous_stride();

    SmallVector<LogicalTensorDesc> out_descs(1u);
    out_descs[0] = {dst_layout, inputs[0].comp_node};
    return {out_descs, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& matmul = def.cast_final_safe<MatrixMul>();
    auto&& cn = inputs[0]->comp_node();

    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds(inputs.size());
    TensorLayout layout1 = inputs[0]->layout(), layout2 = inputs[1]->layout();

    DnnOprCaller<megdnn::MatrixMul> dnn_opr(cn, matmul.param(), matmul.policy());

    if (matmul.dimA == matmul.dimB && matmul.dimB >= 3) {  // only happens in backward
        for (size_t i = 1; i + 1 < layout1.ndim; ++i) {
            layout1[0] *= layout1[i];
            layout2[0] *= layout2[i];
        }
        layout1[1] = layout1[layout1.ndim - 1];
        layout1.ndim = 2;
        layout1.init_contiguous_stride();
        layout2[1] = layout2[layout2.ndim - 1];
        layout2.ndim = 2;
        layout2.init_contiguous_stride();
    }

    DType dst_dtype;
    dnn_opr.op()->deduce_dtype(layout1.dtype, layout1.dtype, dst_dtype);

    // only matters when layout1 has dim 2
    if (matmul.transposeA)
        std::swap(layout1.shape[0], layout1.shape[1]);
    // only matters when layout2 has dim 2
    if (matmul.transposeB)
        std::swap(layout2.shape[0], layout2.shape[1]);

    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;
    TensorLayout real_dst_layout(dst_dtype);
    if (validated) {
        real_dst_layout = output_descs[0].layout;
    } else {
        size_t ri = 0;
        for (size_t i = 0; i < dim1 - 2; i++)
            real_dst_layout[ri++] = layout1[i];
        real_dst_layout[ri++] = layout1[dim1 - 2];
        if (dim2 == 2)
            real_dst_layout[ri++] = layout2[dim2 - 1];
        real_dst_layout.ndim = ri;
        real_dst_layout.init_contiguous_stride();
    }

    if (dim1 == 0 || dim2 == 0 || layout1[layout1.ndim - 1] == 0) {
        auto out = Tensor::make(real_dst_layout, cn);

        if (!out->empty()) {
            dev_tensor_memset(out->dev_tensor(), 0);
        }
        return {out};
    }

    TensorLayout layout_a = layout1, layout_b = layout2;
    if (dim1 > 2) {
        size_t batch = std::accumulate(
                layout1.shape, layout1.shape + dim1 - 1, (size_t)1,
                std::multiplies<size_t>());

        TensorShape na = TensorShape{batch, layout1[dim1 - 1]};
        auto inp1 = inputs[0];
        if (!layout1.try_reshape(layout_a, na)) {
            inp1 = Tensor::make(inp1->blob(), inp1->offset(), layout1);
            inp1->to_contiguous_inplace();
            layout1 = inp1->layout();
            layout_a = TensorLayout{{batch, layout1[dim1 - 1]}, layout1.dtype};
        }

        layout_a.init_contiguous_stride();
        inp_tensornds[0] = inp1->dnn_tensor();
        inp_tensornds[0].layout = layout_a;
    } else {
        inp_tensornds[0] = inputs[0]->dnn_tensor();
    }

    inp_tensornds[1] = inputs[1]->dnn_tensor();

    TensorLayout dst_layout = TensorLayout({layout_a[0], layout_b[1]}, dst_dtype);
    dst_layout.init_contiguous_stride();

    if (matmul.transposeA)
        std::swap(layout_a.shape[0], layout_a.shape[1]);
    if (matmul.transposeB)
        std::swap(layout_b.shape[0], layout_b.shape[1]);

    if (matmul.dimA == matmul.dimB && matmul.dimB >= 3) {  // only happens in backward
        inp_tensornds[0].layout = layout_a;
        inp_tensornds[1].layout = layout_b;
    }
    auto out = Tensor::make(dst_layout, cn);
    dnn_opr.exec_fastrun(inp_tensornds[0], inp_tensornds[1], out);
    return {out->sub(0, real_dst_layout)};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = layout_checker[1] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(MatrixMul, MatrixMul)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .get_input_layout_constraint(get_input_layout_constraint)
        .fallback();
}  // namespace matrix_mul
}  // namespace

namespace {
namespace batched_matrix_mul {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& matmul = def.cast_final_safe<BatchedMatrixMul>();
    mgb_assert(inputs.size() == 2);
    auto inp1 = SymbolVar{inputs[0]}, inp2 = SymbolVar{inputs[1]};
    auto dim1 = matmul.dimA, dim2 = matmul.dimB;

    auto cn = inputs[0]->comp_node();
    using IndexDesc = opr::Subtensor::IndexDesc;
    OperatorNodeConfig config{matmul.make_name(), cn};

    DTypeScalar vi{-2};
    auto graph = inputs[0]->owner_graph();
    auto idx = opr::ImmutableTensor::make(*graph, vi, config);

    auto shp1 = inp1.symshape();
    auto shp2 = inp2.symshape();
    SymbolVar shp1_head, shp1_tail, shp2_head, shp2_tail;
    SymbolVar batch_shape;
    if (dim1 > dim2) {
        HostTensorND hv = HostTensorND(cn, {1}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = -dim2;
        IndexDesc head_desc(1);
        head_desc[0].end = opr::ImmutableTensor::make(*graph, hv, config);
        shp1_head = opr::Subtensor::make(shp1, head_desc);
        shp2 = opr::Concat::make({shp1_head, shp2}, 0, cn);
        inp2 = inp2.broadcast(shp2);
        head_desc[0].end = idx;
        batch_shape = opr::Subtensor::make(shp1, head_desc);
    }
    if (dim2 > dim1) {
        HostTensorND hv = HostTensorND(cn, {1}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = -dim1;
        IndexDesc head_desc(1);
        head_desc[0].end = opr::ImmutableTensor::make(*graph, hv, config);
        shp2_head = opr::Subtensor::make(shp2, head_desc);
        shp1 = opr::Concat::make({shp2_head, shp1}, 0, cn);
        inp1 = inp1.broadcast(shp1);
        head_desc[0].end = idx;
        batch_shape = opr::Subtensor::make(shp2, head_desc);
    }
    if (dim1 == dim2) {
        IndexDesc head_desc(1);
        head_desc[0].end = idx;
        batch_shape = opr::Subtensor::make(shp1, head_desc);
    }

    auto maxdim = dim1 > dim2 ? dim1 : dim2;
    if (maxdim > 3) {
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        shp1_tail = opr::Subtensor::make(shp1, tail_desc);
        auto batch = opr::Reduce::make(batch_shape, {Reduce::Mode::PRODUCT, 0});
        shp1 = opr::Concat::make({batch, shp1_tail}, 0, cn);
        inp1 = inp1.reshape(shp1);
        shp2_tail = opr::Subtensor::make(shp2, tail_desc);
        shp2 = opr::Concat::make({batch, shp2_tail}, 0, cn);
        inp2 = inp2.reshape(shp2);
    }

    auto result = opr::BatchedMatrixMul::make(
            inp1, inp2, matmul.param(), matmul.policy(), config);

    if (maxdim > 3) {
        auto result_shp = result.symshape();
        IndexDesc tail_desc(1);
        tail_desc[0].begin = idx;
        auto shp_tail = opr::Subtensor::make(result_shp, tail_desc);
        result_shp = opr::Concat::make({batch_shape, shp_tail}, 0, cn);
        result = result.reshape(result_shp);
    }
    return result;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& matmul = def.cast_final_safe<BatchedMatrixMul>();
    TensorLayout layout1 = inputs[0].layout, layout2 = inputs[1].layout;
    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;

    DType dst_dtype;

    DnnOprHelper<megdnn::MatrixMul> dnn_opr(matmul.param());
    dnn_opr.opr().deduce_dtype(layout1.dtype, layout1.dtype, dst_dtype);

    if (dim1 == 0 || dim2 == 0) {
        return {{{TensorLayout(dst_dtype), inputs[0].comp_node}}, false};
    }

    if (matmul.transposeA)
        std::swap(layout1[dim1 - 1], layout1[dim1 - 2]);
    if (matmul.transposeB)
        std::swap(layout2[dim2 - 1], layout2[dim2 - 2]);

    TensorLayout dst_layout(dst_dtype);
    size_t di = 0;
    if (dim1 > dim2) {
        for (size_t i = 0; i < dim1 - 2; i++)
            dst_layout[di++] = layout1[i];
    } else {
        for (size_t i = 0; i < dim2 - 2; i++)
            dst_layout[di++] = layout2[i];
    }
    if (dim1 > 1)
        dst_layout[di++] = layout1[dim1 - 2];
    if (dim2 > 1)
        dst_layout[di++] = layout2[dim2 - 1];
    dst_layout.ndim = di;
    dst_layout.init_contiguous_stride();

    SmallVector<LogicalTensorDesc> out_descs(1u);
    out_descs[0] = {dst_layout, inputs[0].comp_node};
    return {out_descs, true};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& matmul = def.cast_final_safe<BatchedMatrixMul>();
    auto&& cn = inputs[0]->comp_node();

    TensorLayout layout1 = inputs[0]->layout(), layout2 = inputs[1]->layout();
    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;

    DnnOprCaller<megdnn::BatchedMatrixMul> dnn_opr(cn, matmul.param(), matmul.policy());
    DType dst_dtype;
    dnn_opr.op()->deduce_dtype(layout1.dtype, layout1.dtype, dst_dtype);

    TensorShape tshp, batch_shp;
    size_t j = 0;
    auto inp1 = inputs[0], inp2 = inputs[1];
    if (dim1 > dim2) {
        for (size_t i = 0; i < dim1 - 2; i++)
            tshp[j++] = layout1.shape[i];
        batch_shp = tshp;
        batch_shp.ndim = dim1 - 2;
        tshp[j++] = layout2[layout2.ndim - 2];
        tshp[j++] = layout2[layout2.ndim - 1];
        tshp.ndim = j;
        layout2 = layout2.broadcast(tshp);
    }
    if (dim2 > dim1) {
        for (size_t i = 0; i < dim2 - 2; i++)
            tshp[j++] = layout2.shape[i];
        batch_shp = tshp;
        batch_shp.ndim = dim2 - 2;
        tshp[j++] = layout1[layout1.ndim - 2];
        tshp[j++] = layout1[layout1.ndim - 1];
        tshp.ndim = j;
        layout1 = layout1.broadcast(tshp);
    }
    if (dim1 == dim2) {
        for (size_t i = 0; i < dim1 - 2; i++)
            tshp[j++] = layout1.shape[i];
        batch_shp = tshp;
        batch_shp.ndim = dim1 - 2;
    }

    TensorShape shp1 = batch_shp, shp2 = batch_shp;
    shp1.ndim += 2;
    shp2.ndim += 2;
    size_t maxdim = dim1 > dim2 ? dim1 : dim2;
    size_t nbatch = batch_shp[0];
    if (maxdim > 3) {
        nbatch = std::accumulate(
                batch_shp.shape, batch_shp.shape + batch_shp.ndim, (size_t)1,
                std::multiplies<size_t>());

        TensorLayout layout_a;

        // batched_matmul does not support memory forwarding, so ensure contiguous
        // manually
        TensorShape nl1 = TensorShape(
                {nbatch, layout1[layout1.ndim - 2], layout1[layout1.ndim - 1]});
        inp1 = Tensor::make(inputs[0]->blob(), inputs[0]->offset(), layout1);
        inp1->to_contiguous_inplace();
        layout1 = inp1->layout();
        layout_a = layout1.reshape(nl1);
        layout1 = layout_a;

        TensorShape nl2 = TensorShape(
                {nbatch, layout2[layout2.ndim - 2], layout2[layout2.ndim - 1]});
        inp2 = Tensor::make(inputs[1]->blob(), inputs[1]->offset(), layout2);
        inp2->to_contiguous_inplace();
        layout2 = inp2->layout();
        layout_a = layout2.reshape(nl2);
        layout2 = layout_a;
    }

    TensorLayout dst_layout(
            {nbatch, matmul.transposeA ? layout1[2] : layout1[1],
             matmul.transposeB ? layout2[1] : layout2[2]},
            dst_dtype);
    dst_layout.init_contiguous_stride();

    if (dim1 == 0 || dim2 == 0 || layout1[layout1.ndim - 1] == 0) {
        auto out = Tensor::make(dst_layout, cn);

        if (!out->empty()) {
            dev_tensor_memset(out->dev_tensor(), 0);
        }
        return {out};
    }

    SmallVector<megdnn::TensorND> inp_tensornds(2u);
    inp_tensornds[0] = inp1->dnn_tensor();
    inp_tensornds[0].layout = layout1;
    inp_tensornds[1] = inp2->dnn_tensor();
    inp_tensornds[1].layout = layout2;

    auto out = Tensor::make(dst_layout, cn);

    dnn_opr.exec_fastrun(inp_tensornds[0], inp_tensornds[1], out);

    shp1[shp1.ndim - 2] = dst_layout[dst_layout.ndim - 2];
    shp1[shp1.ndim - 1] = dst_layout[dst_layout.ndim - 1];
    if (maxdim > 3) {
        dst_layout = dst_layout.reshape(shp1);
    }
    return {out->sub(0, dst_layout)};
}

SmallVector<VarNode::LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<VarNode::LayoutConstraintCallback> layout_checker(inputs.size());
    layout_checker[0] = layout_checker[1] = [](const TensorLayout& layout) {
        return layout.is_contiguous();
    };
    return layout_checker;
}

OP_TRAIT_REG(BatchedMatrixMul, BatchedMatrixMul)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .get_input_layout_constraint(get_input_layout_constraint)
        .apply_on_physical_tensor(apply_on_physical_tensor)
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

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto comp_node = inputs[0]->comp_node();
    using TensorND = megdnn::TensorND;
    SmallVector<TensorND> inp_tensornds;
    inp_tensornds.reserve(inputs.size());
    DnnOprCaller<megdnn::Dot> dnn_opr(comp_node);
    for (unsigned i = 0; i < inputs.size(); ++i) {
        auto dnn_ten = inputs[i]->dnn_tensor();
        inp_tensornds.push_back(dnn_ten);
    }
    TensorLayout oup_layout{inputs[0]->dtype()};
    auto inp1_tensor = inputs[0]->dnn_tensor();
    auto inp2_tensor = inputs[1]->dnn_tensor();
    oup_layout = dnn_opr.deduce_layout(inp1_tensor.layout, inp2_tensor.layout);

    if (inputs[0]->layout().is_empty() || inputs[1]->layout().is_empty()) {
        auto out = Tensor::make(oup_layout, comp_node);
        if (!out->empty()) {
            dev_tensor_memset(out->dev_tensor(), 0);
        }
        return {out};
    }

    auto out = Tensor::make(oup_layout, comp_node);
    dnn_opr.exec_with_ws(inp_tensornds[0], inp_tensornds[1], out);

    return {out};
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    mgb_assert(
            inputs.size() == 2, "Dot expects 2 inputs; got %lu actually",
            inputs.size());
    SmallVector<LogicalTensorDesc> dests(1);
    dests[0].layout = TensorLayout(TensorShape{1}, inputs[0].layout.dtype);
    dests[0].comp_node = inputs[0].comp_node;
    bool validated = inputs[0].layout.ndim != 0 && inputs[1].layout.ndim != 0;
    return {dests, validated};
}

OP_TRAIT_REG(Dot, Dot, mgb::opr::Dot)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace dot
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb
