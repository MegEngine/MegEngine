#include <numeric>
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/blas.h"

#include "../algo_chooser.h"

namespace mgb {
namespace imperative {

namespace {
namespace matrix_mul {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& matmul = def.cast_final_safe<MatrixMul>();
    mgb_assert(inputs.size() == 2);
    OperatorNodeConfig config{matmul.make_name()};
    return opr::MatrixMul::make(
            inputs[0], inputs[1], matmul.param(), matmul.policy(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& matmul = def.cast_final_safe<MatrixMul>();
    auto layout1 = inputs[0].layout;
    auto layout2 = inputs[1].layout;
    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;

    if (dim1 == 0 || dim2 == 0) {
        return {{{TensorLayout(layout1.dtype), inputs[0].comp_node}}, false};
    }

    if (matmul.transposeA)
        std::swap(layout1[0], layout1[1]);
    if (matmul.transposeB)
        std::swap(layout2[0], layout2[1]);

    mgb_assert(layout1[dim1 - 1] == layout2[0]);
    TensorLayout dst_layout(layout1.dtype);
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

    // only matters when layout1 has dim 2
    if (matmul.transposeA)
        std::swap(layout1.shape[0], layout1.shape[1]);
    // only matters when layout2 has dim 2
    if (matmul.transposeB)
        std::swap(layout2.shape[0], layout2.shape[1]);

    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;
    TensorLayout real_dst_layout(layout1.dtype);
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
        DeviceTensorND out =
                BlobManager::inst()->alloc_workspace_with_defrag(cn, real_dst_layout);
        if (!out.empty()) {
            dev_tensor_memset(out, 0);
        }
        return {Tensor::make(out)};
    }

    TensorLayout layout_a = layout1, layout_b = layout2;
    if (dim1 == 1) {
        layout_a.add_axis_cont_inplace(0);
        inp_tensornds[0] = inputs[0]->dnn_tensor();
        inp_tensornds[0].layout = layout_a;
    } else if (dim1 > 2) {
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

    if (dim2 == 1) {
        layout_b.add_axis_inplace(1, 1, 1);
        inp_tensornds[1] = inputs[1]->dnn_tensor();
        inp_tensornds[1].layout = layout_b;
    } else {
        inp_tensornds[1] = inputs[1]->dnn_tensor();
    }

    TensorLayout dst_layout = TensorLayout({layout_a[0], layout_b[1]}, layout_a.dtype);
    dst_layout.init_contiguous_stride();

    DnnOprCaller<megdnn::MatrixMul> dnn_opr(cn);
    dnn_opr.op->param() = matmul.param();

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, dst_layout);
    size_t sz = setup_algo<megdnn::MatrixMul>(
            {layout_a, layout_b, dst_layout}, dnn_opr.op.get(), 0, false, false, cn,
            matmul.policy(), false);
    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);

    dnn_opr.op->exec(inp_tensornds[0], inp_tensornds[1], out.as_megdnn(), dnn_wk);
    return {Tensor::make(out.sub(SubTensorSpec::make_from_layout(real_dst_layout)))};
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
    OperatorNodeConfig config{matmul.make_name()};
    return opr::BatchedMatrixMul::make(
            inputs[0], inputs[1], matmul.param(), matmul.policy(), config);
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& matmul = def.cast_final_safe<BatchedMatrixMul>();
    TensorLayout layout1 = inputs[0].layout, layout2 = inputs[1].layout;
    size_t dim1 = layout1.ndim, dim2 = layout2.ndim;

    if (dim1 == 0 || dim2 == 0) {
        return {{{TensorLayout(layout1.dtype), inputs[0].comp_node}}, false};
    }

    if (matmul.transposeA)
        std::swap(layout1[dim1 - 1], layout1[dim1 - 2]);
    if (matmul.transposeB)
        std::swap(layout2[dim2 - 1], layout2[dim2 - 2]);

    TensorLayout dst_layout(layout1.dtype);
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

    bool remove_row = false, remove_col = false;
    if (dim1 == 1) {
        dim1 = 2;
        remove_row = true;
    }
    if (dim2 == 1) {
        dim2 = 2;
        remove_col = true;
    }

    if (remove_row)
        layout1.add_axis_cont_inplace(0);
    if (remove_col)
        layout2.add_axis_inplace(1, 1, 1);

    TensorShape tshp, batch_shp;
    size_t j = 0;
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
    auto inp1 = inputs[0], inp2 = inputs[1];
    if (maxdim > 3) {
        nbatch = std::accumulate(
                batch_shp.shape, batch_shp.shape + batch_shp.ndim, (size_t)1,
                std::multiplies<size_t>());

        TensorLayout layout_a;

        TensorShape nl1 = TensorShape(
                {nbatch, layout1[layout1.ndim - 2], layout1[layout1.ndim - 1]});
        if (!layout1.try_reshape(layout_a, nl1)) {
            inp1 = Tensor::make(inputs[0]->blob(), inputs[0]->offset(), layout1);
            inp1->to_contiguous_inplace();
            layout1 = inp1->layout();
        }
        layout1 = layout_a;

        TensorShape nl2 = TensorShape(
                {nbatch, layout2[layout2.ndim - 2], layout2[layout2.ndim - 1]});
        if (!layout2.try_reshape(layout_a, nl2)) {
            inp2 = Tensor::make(inputs[1]->blob(), inputs[1]->offset(), layout2);
            inp2->to_contiguous_inplace();
            layout2 = inp2->layout();
        }
        layout2 = layout_a;
    }

    TensorLayout dst_layout(
            {nbatch, matmul.transposeA ? layout1[2] : layout1[1],
             matmul.transposeB ? layout2[1] : layout2[2]},
            layout1.dtype);
    dst_layout.init_contiguous_stride();

    if (dim1 == 0 || dim2 == 0 || layout1[layout1.ndim - 1] == 0) {
        DeviceTensorND out =
                BlobManager::inst()->alloc_workspace_with_defrag(cn, dst_layout);
        if (!out.empty()) {
            dev_tensor_memset(out, 0);
        }
        return {Tensor::make(out)};
    }

    using TensorND = megdnn::TensorND;
    TensorND inp_nd1 = inp1->dnn_tensor();
    inp_nd1.layout = layout1;
    TensorND inp_nd2 = inp2->dnn_tensor();
    inp_nd2.layout = layout2;

    DeviceTensorND out =
            BlobManager::inst()->alloc_workspace_with_defrag(cn, dst_layout);

    DnnOprCaller<megdnn::BatchedMatrixMul> dnn_opr(cn);
    dnn_opr.op->param() = matmul.param();

    size_t sz = setup_algo<megdnn::BatchedMatrixMul>(
            {layout1, layout2, dst_layout}, dnn_opr.op.get(), 0, false, false, cn,
            matmul.policy(), false);
    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);
    dnn_opr.op->exec(inp_nd1, inp_nd2, out.as_megdnn(), dnn_wk);

    shp1[shp1.ndim - 2] = dst_layout[dst_layout.ndim - 2];
    shp1[shp1.ndim - 1] = dst_layout[dst_layout.ndim - 1];
    if (maxdim > 3) {
        dst_layout = dst_layout.reshape(shp1);
    }
    if (remove_row) {
        dst_layout = dst_layout.remove_axis(maxdim - 2);
    }
    if (remove_col) {
        dst_layout = dst_layout.remove_axis(maxdim - 1);
    }
    return {Tensor::make(out.sub(SubTensorSpec::make_from_layout(dst_layout)))};
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
    dnn_opr.op->deduce_layout(inp1_tensor.layout, inp2_tensor.layout, oup_layout);

    if (inputs[0]->layout().is_empty() || inputs[1]->layout().is_empty()) {
        DeviceTensorND out =
                BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);
        if (!out.empty()) {
            dev_tensor_memset(out, 0);
        }
        return {Tensor::make(out)};
    }

    auto sz = dnn_opr.op->get_workspace_in_bytes(
            inp_tensornds[0].layout, inp_tensornds[1].layout, output_descs[0].layout);

    DeviceTensorND out_devtensor =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, oup_layout);

    TensorLayout w_layout({sz}, dtype::Byte());
    auto dnn_wk = dnn_opr.create_workspace(w_layout);

    dnn_opr.op->exec(
            inp_tensornds[0], inp_tensornds[1], out_devtensor.as_megdnn(), dnn_wk);

    return {Tensor::make(out_devtensor)};
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
