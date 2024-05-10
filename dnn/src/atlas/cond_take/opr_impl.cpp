#include "opr_impl.h"
#include "aclnnop/aclnn_abs.h"
#include "aclnnop/aclnn_all.h"
#include "aclnnop/aclnn_any.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_ge_scalar.h"
#include "aclnnop/aclnn_gt_scalar.h"
#include "aclnnop/aclnn_le_scalar.h"
#include "aclnnop/aclnn_lt_scalar.h"
#include "aclnnop/aclnn_masked_select.h"
#include "aclnnop/aclnn_nonzero_v2.h"
#include "aclnnop/aclnn_sub.h"
#include "aclnnop/aclnn_unique2.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

CondTakeImpl::Output CondTakeImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in mask, _megdnn_workspace workspace,
        DynOutMallocPolicyCall malloc_policy) {
    check_exec_get_size(data.layout, mask.layout, workspace.size);
    auto handle = concrete_handle(this->handle());

    // convert mask to float mask
    TensorLayout float_mask_layout(TensorShape(mask.layout), dtype::Float32());
    AclMem acl_float_mask_mem(float_mask_layout.span().dist_byte(), handle);
    TensorND float_mask(acl_float_mask_mem.ptr(), float_mask_layout);
    AclTensor acl_float_mask(float_mask);
    auto type_cvt_opr = handle->create_operator<TypeCvt>();
    type_cvt_opr->exec(mask, float_mask);

    // scalar value
    AclScalar acl_scalar_value(param().val);

    // perform abs(mask - value) for eq and neq
    if (param().mode == Param::Mode::EQ || param().mode == Param::Mode::NEQ) {
        // tensor value
        TensorLayout value_layout(TensorShape({1}), dtype::Float32());
        AclMem acl_value_mem(value_layout.span().dist_byte(), handle);
        TensorND value(acl_value_mem.ptr(), value_layout);
        AclTensor acl_value(value);
        auto fill_opr = handle->create_operator<Fill>();
        fill_opr->param().value = param().val;
        fill_opr->exec(value, {});

        AclScalar acl_alpha(1.0);

        uint64_t sub_ws_size = 0;
        aclOpExecutor* sub_executor = nullptr;
        aclnn_check(aclnnInplaceSubGetWorkspaceSize(
                acl_float_mask.get(), acl_value.get(), acl_alpha.get(), &sub_ws_size,
                &sub_executor));
        AclMem sub_ws(sub_ws_size, handle);
        aclnn_check(aclnnInplaceSub(
                sub_ws.ptr(), sub_ws_size, sub_executor, handle->stream()));

        uint64_t abs_ws_size = 0;
        aclOpExecutor* abs_executor = nullptr;
        aclnn_check(aclnnAbsGetWorkspaceSize(
                acl_float_mask.get(), acl_float_mask.get(), &abs_ws_size,
                &abs_executor));
        AclMem abs_ws(abs_ws_size, handle);
        aclnn_check(
                aclnnAbs(abs_ws.ptr(), abs_ws_size, abs_executor, handle->stream()));

        acl_scalar_value = AclScalar(param().eps);
    }

    // perform logic to get bool mask
    TensorLayout bool_mask_layout({TensorShape(mask.layout), dtype::Bool()});
    AclMem acl_bool_mask_mem(bool_mask_layout.span().dist_byte(), handle);
    TensorND bool_mask(acl_bool_mask_mem.ptr(), bool_mask_layout);
    AclTensor acl_bool_mask(bool_mask);
    switch (param().mode) {
        case Param::Mode::EQ: {
            uint64_t eq_ws_size = 0;
            aclOpExecutor* eq_executor = nullptr;
            aclnn_check(aclnnLtScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &eq_ws_size, &eq_executor));
            AclMem eq_ws(eq_ws_size, handle);
            aclnn_check(aclnnLtScalar(
                    eq_ws.ptr(), eq_ws_size, eq_executor, handle->stream()));
            break;
        }
        case Param::Mode::NEQ: {
            uint64_t neq_ws_size = 0;
            aclOpExecutor* neq_executor = nullptr;
            aclnn_check(aclnnGeScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &neq_ws_size, &neq_executor));
            AclMem neq_ws(neq_ws_size, handle);
            aclnn_check(aclnnGeScalar(
                    neq_ws.ptr(), neq_ws_size, neq_executor, handle->stream()));
            break;
        }
        case Param::Mode::LT: {
            uint64_t lt_ws_size = 0;
            aclOpExecutor* lt_executor = nullptr;
            aclnn_check(aclnnLtScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &lt_ws_size, &lt_executor));
            AclMem lt_ws(lt_ws_size, handle);
            aclnn_check(aclnnLtScalar(
                    lt_ws.ptr(), lt_ws_size, lt_executor, handle->stream()));
            break;
        }
        case Param::Mode::GT: {
            uint64_t gt_ws_size = 0;
            aclOpExecutor* gt_executor = nullptr;
            aclnn_check(aclnnGtScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &gt_ws_size, &gt_executor));
            AclMem gt_ws(gt_ws_size, handle);
            aclnn_check(aclnnGtScalar(
                    gt_ws.ptr(), gt_ws_size, gt_executor, handle->stream()));
            break;
        }
        case Param::Mode::LEQ: {
            uint64_t le_ws_size = 0;
            aclOpExecutor* le_executor = nullptr;
            aclnn_check(aclnnLeScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &le_ws_size, &le_executor));
            AclMem le_ws(le_ws_size, handle);
            aclnn_check(aclnnLeScalar(
                    le_ws.ptr(), le_ws_size, le_executor, handle->stream()));
            break;
        }
        case Param::Mode::GEQ: {
            uint64_t ge_ws_size = 0;
            aclOpExecutor* ge_executor = nullptr;
            aclnn_check(aclnnGeScalarGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_bool_mask.get(),
                    &ge_ws_size, &ge_executor));
            AclMem ge_ws(ge_ws_size, handle);
            aclnn_check(aclnnGeScalar(
                    ge_ws.ptr(), ge_ws_size, ge_executor, handle->stream()));
            break;
        }
        default:
            megdnn_throw("error mode");
            break;
    }

    // get num of true elements in bool mask
    TensorLayout collapse_2dim_bool_mask_layout(
            TensorShape({1, bool_mask.layout.total_nr_elems()}), dtype::Bool());
    TensorND collapse_2dim_bool_mask(
            bool_mask.raw_ptr(), collapse_2dim_bool_mask_layout);
    AclTensor acl_collapse_2dim_bool_mask(collapse_2dim_bool_mask);

    TensorLayout any_value_bool_mask_layout({1}, dtype::Bool());
    AclMem any_value_bool_mask_mem(
            any_value_bool_mask_layout.span().dist_byte(), handle);
    TensorND any_value_bool_mask(
            any_value_bool_mask_mem.ptr(), any_value_bool_mask_layout);
    AclTensor acl_any_value_bool_mask(any_value_bool_mask);

    std::vector<int64_t> any_dim(1, 1);
    AclIntArray acl_any_dim(any_dim.data(), 1);

    uint64_t any_ws_size = 0;
    aclOpExecutor* any_executor = nullptr;
    aclnn_check(aclnnAnyGetWorkspaceSize(
            acl_collapse_2dim_bool_mask.get(), acl_any_dim.get(), false,
            acl_any_value_bool_mask.get(), &any_ws_size, &any_executor));
    AclMem any_ws(any_ws_size, handle);
    aclnn_check(aclnnAny(any_ws.ptr(), any_ws_size, any_executor, handle->stream()));

    bool any_value_bool_mask_host = false;
    acl_check(aclrtMemcpyAsync(
            &any_value_bool_mask_host, sizeof(bool), any_value_bool_mask.raw_ptr(),
            sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST, handle->stream()));
    acl_check(aclrtSynchronizeStream(handle->stream()));

    if (!any_value_bool_mask_host) {
        auto out_data = malloc_policy.alloc_output(0, data.layout.dtype, {0});
        auto out_idx = malloc_policy.alloc_output(1, dtype::Int32(), {0});
        return {{out_data, out_idx}};
    }

    TensorLayout all_value_bool_mask_layout({1}, dtype::Bool());
    AclMem all_value_bool_mask_mem(
            all_value_bool_mask_layout.span().dist_byte(), handle);
    TensorND all_value_bool_mask(
            all_value_bool_mask_mem.ptr(), all_value_bool_mask_layout);
    AclTensor acl_all_value_bool_mask(all_value_bool_mask);

    std::vector<int64_t> all_dim(1, 1);
    AclIntArray acl_all_dim(all_dim.data(), 1);

    uint64_t all_ws_size = 0;
    aclOpExecutor* all_executor = nullptr;
    aclnn_check(aclnnAllGetWorkspaceSize(
            acl_collapse_2dim_bool_mask.get(), acl_all_dim.get(), false,
            acl_all_value_bool_mask.get(), &all_ws_size, &all_executor));
    AclMem all_ws(all_ws_size, handle);
    aclnn_check(aclnnAll(all_ws.ptr(), all_ws_size, all_executor, handle->stream()));

    bool all_value_bool_mask_host = false;
    acl_check(aclrtMemcpyAsync(
            &all_value_bool_mask_host, sizeof(bool), all_value_bool_mask.raw_ptr(),
            sizeof(bool), ACL_MEMCPY_DEVICE_TO_HOST, handle->stream()));
    acl_check(aclrtSynchronizeStream(handle->stream()));

    int64_t num_true = 0;
    if (all_value_bool_mask_host) {
        num_true = bool_mask_layout.total_nr_elems();
    } else {
        TensorLayout unique_value_out_layout({2}, dtype::Bool());
        AclMem unique_value_out_mem(unique_value_out_layout.span().dist_byte(), handle);
        TensorND unique_value_out(unique_value_out_mem.ptr(), unique_value_out_layout);
        AclTensor acl_unique_value_out(unique_value_out);

        TensorLayout unique_count_out_layout({2}, dtype::Complex64());
        AclMem unique_count_out_mem(unique_count_out_layout.span().dist_byte(), handle);
        TensorND unique_count_out(unique_count_out_mem.ptr(), unique_count_out_layout);
        AclTensor acl_unique_count_out(unique_count_out, ACL_FORMAT_ND, ACL_INT64);

        TensorLayout unique_inverse_out_layout(
                TensorShape(bool_mask.layout), dtype::Complex64());
        AclMem unique_inverse_out_mem(
                unique_inverse_out_layout.span().dist_byte(), handle);
        TensorND unique_inverse_out(
                unique_inverse_out_mem.ptr(), unique_inverse_out_layout);
        AclTensor acl_unique_inverse_out(unique_inverse_out, ACL_FORMAT_ND, ACL_INT64);

        uint64_t unique2_ws_size = 0;
        aclOpExecutor* unique2_executor = nullptr;
        aclnn_check(aclnnUnique2GetWorkspaceSize(
                acl_bool_mask.get(), true, false, true, acl_unique_value_out.get(),
                acl_unique_inverse_out.get(), acl_unique_count_out.get(),
                &unique2_ws_size, &unique2_executor));
        AclMem unique2_ws(unique2_ws_size, handle);
        aclnn_check(aclnnUnique2(
                unique2_ws.ptr(), unique2_ws_size, unique2_executor, handle->stream()));

        acl_check(aclrtMemcpyAsync(
                &num_true, sizeof(int64_t),
                static_cast<void*>(
                        static_cast<uint8_t*>(unique_count_out_mem.ptr()) +
                        sizeof(int64_t)),
                sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST, handle->stream()));
        acl_check(aclrtSynchronizeStream(handle->stream()));
    }

    size_t out_size = static_cast<size_t>(num_true);
    auto out_data = malloc_policy.alloc_output(0, data.layout.dtype, {out_size});
    auto out_idx = malloc_policy.alloc_output(1, dtype::Int32(), {out_size});
    if (num_true <= 0) {
        return {{out_data, out_idx}};
    }

    // perform masked select to get fake out
    // fake out
    size_t fake_out_size = data.layout.dtype.size(data.layout.total_nr_elems());
    AclMem fake_out_mem(fake_out_size, handle);
    TensorLayout fake_out_layout({data.layout.total_nr_elems()}, data.layout.dtype);
    TensorND fake_out(fake_out_mem.ptr(), fake_out_layout);
    AclTensor acl_fake_out(fake_out);

    AclTensor acl_data(data);
    uint64_t masked_select_ws_size = 0;
    aclOpExecutor* masked_select_executor = nullptr;
    aclnnMaskedSelectGetWorkspaceSize(
            acl_data.get(), acl_bool_mask.get(), acl_fake_out.get(),
            &masked_select_ws_size, &masked_select_executor);
    AclMem masked_select_ws(masked_select_ws_size, handle);
    aclnn_check(aclnnMaskedSelect(
            masked_select_ws.ptr(), masked_select_ws_size, masked_select_executor,
            handle->stream()));

    // copy num of ture elems to real out
    aclrtMemcpyAsync(
            out_data.raw_ptr(), data.layout.dtype.size(out_size), fake_out_mem.ptr(),
            data.layout.dtype.size(out_size), ACL_MEMCPY_DEVICE_TO_DEVICE,
            handle->stream());

    // get out idx
    // collapse bool_mask
    TensorND collapse_bool_mask(
            acl_bool_mask_mem.ptr(), bool_mask_layout.collapse_contiguous());
    AclTensor acl_collapse_bool_mask(collapse_bool_mask);

    // get int64 idx
    TensorLayout int64_out_idx_layout(
            TensorShape({static_cast<size_t>(num_true)}), dtype::Complex64());
    AclMem int64_out_idx_mem(int64_out_idx_layout.span().dist_byte(), handle);
    TensorND int64_out_idx(int64_out_idx_mem.ptr(), int64_out_idx_layout);
    AclTensor acl_int64_out_idx(int64_out_idx, ACL_FORMAT_ND, ACL_INT64);

    uint64_t non_zero_v2_ws_size = 0;
    aclOpExecutor* non_zero_v2_executor = nullptr;
    aclnn_check(aclnnNonzeroV2GetWorkspaceSize(
            acl_collapse_bool_mask.get(), acl_int64_out_idx.get(), &non_zero_v2_ws_size,
            &non_zero_v2_executor));
    AclMem non_zero_v2_ws(non_zero_v2_ws_size, handle);
    aclnn_check(aclnnNonzeroV2(
            non_zero_v2_ws.ptr(), non_zero_v2_ws_size, non_zero_v2_executor,
            handle->stream()));

    // convert int64 idx to int32 idx
    AclTensor acl_out_idx(out_idx),
            acl_non_zero_ret(int64_out_idx, ACL_FORMAT_ND, ACL_INT64);
    uint64_t cast_ws_size = 0;
    aclOpExecutor* cast_executor = nullptr;
    aclnn_check(aclnnCastGetWorkspaceSize(
            acl_non_zero_ret.get(), ACL_INT32, acl_out_idx.get(), &cast_ws_size,
            &cast_executor));
    AclMem cast_ws(cast_ws_size, handle);
    aclnn_check(
            aclnnCast(cast_ws.ptr(), cast_ws_size, cast_executor, handle->stream()));
    return {{out_data, out_idx}};
}