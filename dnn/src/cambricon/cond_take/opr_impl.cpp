#include "src/cambricon/cond_take/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_types.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace cambricon;

namespace {

using Param = CondTake::Param;
using ShapeInfo = std::vector<size_t>;

cnnlLogicOp_t get_logic_op(Param::Mode mode) {
    switch (mode) {
        case Param::Mode::EQ:
            return CNNL_LOGIC_OP_EQ;
        case Param::Mode::NEQ:
            return CNNL_LOGIC_OP_NE;
        case Param::Mode::LT:
            return CNNL_LOGIC_OP_LT;
        case Param::Mode::LEQ:
            return CNNL_LOGIC_OP_LE;
        case Param::Mode::GT:
            return CNNL_LOGIC_OP_GT;
        case Param::Mode::GEQ:
            return CNNL_LOGIC_OP_GE;
        default:
            megdnn_assert(false);
    }
}

}  // namespace
WorkspaceBundle CondTakeImpl::make_bundle(
        const TensorLayout& data, const TensorLayout& mask) {
    auto handle = concrete_handle(this->handle());
    size_t elem_size = mask.total_nr_elems();
    size_t assign_sub_workspace = 0, logicop_workspace = 0, num_true_workspace = 0,
           where_workspace = 0;
    CnnlTensorDescriptor num_true_desc, mask_desc, one_elem_desc;
    ShapeInfo one_elem_shape = {1};
    num_true_desc.set(1, one_elem_shape, CNNL_DTYPE_INT32, CNNL_LAYOUT_ARRAY);
    mask_desc.set(mask.ndim, mask.shape, CNNL_DTYPE_FLOAT, CNNL_LAYOUT_ARRAY);
    one_elem_desc.set(1, one_elem_shape, CNNL_DTYPE_FLOAT, CNNL_LAYOUT_ARRAY);
    cnnl_check(cnnlGetAssignSubWorkspaceSize(
            handle->cnnl_handle(), one_elem_desc.desc(), mask_desc.desc(),
            &assign_sub_workspace));
    cnnl_check(cnnlGetLogicOpWorkspaceSize(
            handle->cnnl_handle(), mask_desc.desc(), one_elem_desc.desc(),
            mask_desc.desc(), &logicop_workspace));
    cnnl_check(cnnlGetNumTrueWorkspaceSize_v2(
            handle->cnnl_handle(), num_true_desc.desc(), &num_true_workspace));
    cnnl_check(cnnlGetWhereWorkspaceSize(
            handle->cnnl_handle(), num_true_desc.desc(), &where_workspace));
    return {nullptr,
            {/*float_mask*/ elem_size * sizeof(float),
             /*assign_sub, logic, num_true*/ 1 * sizeof(float), assign_sub_workspace,
             logicop_workspace, num_true_workspace, where_workspace},
            handle->alignment_requirement()};
}

size_t CondTakeImpl::get_workspace_in_bytes(
        const TensorLayout& data, const TensorLayout& mask) {
    return make_bundle(data, mask).total_size_in_bytes();
}

CondTakeImpl::Output CondTakeImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in mask, _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
    check_exec_get_size(data.layout, mask.layout, workspace.size);
    auto handle = concrete_handle(this->handle());

    auto wk_bundle = make_bundle(data.layout, mask.layout);
    wk_bundle.set(workspace.raw_ptr);

    // float_mask = mask.astype(float32)
    void* float_mask = wk_bundle.get(0);
    TensorLayout float_mask_layout(TensorShape(mask.layout), dtype::Float32());
    megdnn_assert(wk_bundle.get_size(0) >= float_mask_layout.access_bytes());
    auto astype_opr = handle->create_operator<TypeCvt>();
    astype_opr->exec(mask, {float_mask, float_mask_layout});

    void* one_elem = wk_bundle.get(1);
    TensorLayout one_elem_layout(TensorShape({1}), dtype::Float32());

    auto fill_opr = handle->create_operator<Fill>();
    fill_opr->param().value = param().val;
    fill_opr->exec({one_elem, one_elem_layout}, {});

    CnnlTensorDescriptor mask_desc, val_desc;
    ShapeInfo one_elem_shape = {1};
    val_desc.set(1, one_elem_shape, CNNL_DTYPE_FLOAT, CNNL_LAYOUT_ARRAY);
    mask_desc.set(
            float_mask_layout.ndim, float_mask_layout.shape, CNNL_DTYPE_FLOAT,
            CNNL_LAYOUT_ARRAY);
    float alpha = 1.f, beta = 1.f;
    if (param().mode == Param::Mode::EQ || param().mode == Param::Mode::NEQ) {
        cnnl_check(cnnlAssignSub(
                handle->cnnl_handle(), &alpha, val_desc.desc(), one_elem,
                wk_bundle.get(2), wk_bundle.get_size(2), &beta, mask_desc.desc(),
                float_mask));
        cnnl_check(
                cnnlAbs(handle->cnnl_handle(), mask_desc.desc(), float_mask,
                        mask_desc.desc(), float_mask));
        fill_opr->param().value = param().eps;
        fill_opr->exec({one_elem, one_elem_layout}, {});
    }
    cnnlLogicOp_t op = get_logic_op(param().mode);
    if (param().mode == Param::Mode::EQ) {
        op = get_logic_op(Param::Mode::LT);
    }
    if (param().mode == Param::Mode::NEQ) {
        op = get_logic_op(Param::Mode::GEQ);
    }

    cnnl_check(cnnlLogicOp(
            handle->cnnl_handle(), op, mask_desc.desc(), float_mask, val_desc.desc(),
            one_elem, wk_bundle.get(3), wk_bundle.get_size(3), mask_desc.desc(),
            float_mask));

    void* num_true_ptr = wk_bundle.get(1);
    CnnlTensorDescriptor num_true_desc;
    num_true_desc.set(1, one_elem_shape, CNNL_DTYPE_INT32, CNNL_LAYOUT_ARRAY);
    cnnl_check(cnnlNumTrue_v3(
            handle->cnnl_handle(), mask_desc.desc(), float_mask, wk_bundle.get(4),
            wk_bundle.get_size(4), num_true_desc.desc(), num_true_ptr));

    // num_nozero = nozero(mask)
    int32_t num_true = 0;
    cnrt_check(cnrtMemcpyAsync(
            &num_true, num_true_ptr, sizeof(int32_t), handle->queue(),
            cnrtMemcpyDevToHost));
    cnrt_check(cnrtQueueSync(handle->queue()));

    size_t out_size = num_true;
    auto out_data = malloc_policy.alloc_output(0, data.layout.dtype, {out_size});
    auto out_idx = malloc_policy.alloc_output(1, dtype::Int32(), {out_size});

    if (num_true <= 0) {
        return {{out_data, out_idx}};
    }
    // idx = where(mask, num_nozero)
    // out = data[idx]
    CnnlTensorDescriptor index_desc, data_desc, out_desc, mask_desc_flat;
    ShapeInfo dst_shape = {out_size};
    ShapeInfo data_shape = {float_mask_layout.total_nr_elems()};
    index_desc.set(1, dst_shape, CNNL_DTYPE_INT32, CNNL_LAYOUT_ARRAY);
    out_desc.set(
            1, dst_shape, convert_to_cnnl_datatype(data.layout.dtype.enumv()),
            CNNL_LAYOUT_ARRAY);
    data_desc.set(
            1, data_shape, convert_to_cnnl_datatype(data.layout.dtype.enumv()),
            CNNL_LAYOUT_ARRAY);
    mask_desc_flat.set(1, data_shape, CNNL_DTYPE_FLOAT, CNNL_LAYOUT_ARRAY);

    cnnl_check(cnnlWhere_v2(
            handle->cnnl_handle(), mask_desc_flat.desc(), float_mask,
            num_true_desc.desc(), num_true_ptr, false, wk_bundle.get(5),
            wk_bundle.get_size(5), index_desc.desc(), out_idx.raw_ptr()));

    cnnl_check(cnnlIndexSelect(
            handle->cnnl_handle(), 0, data_desc.desc(), data.raw_ptr(),
            index_desc.desc(), out_idx.raw_ptr(), out_desc.desc(), out_data.raw_ptr()));
    return {{out_data, out_idx}};
}

// vim: syntax=cpp.doxygen
