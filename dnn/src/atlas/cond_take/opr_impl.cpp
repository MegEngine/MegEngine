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
#include "aclnnop/aclnn_reduce_sum.h"
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

    TensorLayout bool_mask_layout({TensorShape(mask.layout), dtype::Bool()});
    AclMem acl_bool_mask_mem(bool_mask_layout.span().dist_byte(), handle);
    TensorND bool_mask(acl_bool_mask_mem.ptr(), bool_mask_layout);
    if (mask.layout.dtype.enumv() == DTypeEnum::Bool &&
        (param().mode == Param::Mode::EQ &&
         std::abs(1.0f - param().val) < param().eps)) {
        bool_mask = mask;
    } else {
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
            AclScalar acl_alpha(1.0);
            uint64_t subs_ws_size = 0;
            aclOpExecutor* subs_executor = nullptr;
            aclnn_check(aclnnInplaceSubsGetWorkspaceSize(
                    acl_float_mask.get(), acl_scalar_value.get(), acl_alpha.get(),
                    &subs_ws_size, &subs_executor));
            AclMem subs_ws(subs_ws_size, handle);
            aclnn_check(aclnnInplaceSub(
                    subs_ws.ptr(), subs_ws_size, subs_executor, handle->stream()));

            uint64_t abs_ws_size = 0;
            aclOpExecutor* abs_executor = nullptr;
            aclnn_check(aclnnAbsGetWorkspaceSize(
                    acl_float_mask.get(), acl_float_mask.get(), &abs_ws_size,
                    &abs_executor));
            AclMem abs_ws(abs_ws_size, handle);
            aclnn_check(aclnnAbs(
                    abs_ws.ptr(), abs_ws_size, abs_executor, handle->stream()));

            acl_scalar_value = AclScalar(param().eps);
        }

        // perform logic to get bool mask
        AclTensor acl_bool_mask(bool_mask);
        switch (param().mode) {
            case Param::Mode::EQ: {
                aclnn_call(
                        handle, aclnnLtScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            case Param::Mode::NEQ: {
                aclnn_call(
                        handle, aclnnGeScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            case Param::Mode::LT: {
                aclnn_call(
                        handle, aclnnLtScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            case Param::Mode::GT: {
                aclnn_call(
                        handle, aclnnGtScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            case Param::Mode::LEQ: {
                aclnn_call(
                        handle, aclnnLeScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            case Param::Mode::GEQ: {
                aclnn_call(
                        handle, aclnnGeScalar, acl_float_mask.get(),
                        acl_scalar_value.get(), acl_bool_mask.get());
                break;
            }
            default:
                megdnn_throw("error mode");
                break;
        }
    }

    AclTensor acl_bool_mask(bool_mask);
    // get num of true elements in bool mask
    TensorLayout collapse_2dim_bool_mask_layout(
            TensorShape({1, bool_mask.layout.total_nr_elems()}), dtype::Bool());
    TensorND collapse_2dim_bool_mask(
            bool_mask.raw_ptr(), collapse_2dim_bool_mask_layout);
    AclTensor acl_collapse_2dim_bool_mask(collapse_2dim_bool_mask);

    float num_true = 0;
    AclIntArray acl_dims({1});
    TensorLayout reduce_sum_out_layout(
            TensorShape({static_cast<size_t>(1)}), dtype::Float32());
    AclMem acl_reduce_sum_out_mem(reduce_sum_out_layout.span().dist_byte(), handle);
    TensorND reduce_sum_out(acl_reduce_sum_out_mem.ptr(), reduce_sum_out_layout);
    AclTensor acl_reduce_sum_out(reduce_sum_out);

    uint64_t reduce_sum_ws_size = 0;
    aclOpExecutor* reduce_sum_executor = nullptr;
    aclnnReduceSumGetWorkspaceSize(
            acl_collapse_2dim_bool_mask.get(), acl_dims.get(), false,
            aclDataType::ACL_FLOAT, acl_reduce_sum_out.get(), &reduce_sum_ws_size,
            &reduce_sum_executor);
    AclMem reduce_sum_ws(reduce_sum_ws_size, handle);
    aclnn_check(aclnnReduceSum(
            reduce_sum_ws.ptr(), reduce_sum_ws_size, reduce_sum_executor,
            handle->stream()));
    acl_safe_memcpy_async_with_sync(
            &num_true, sizeof(float), reduce_sum_out.raw_ptr(), sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST, handle->stream());

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
    acl_safe_memcpy_async(
            out_data.raw_ptr(), data.layout.dtype.size(out_size), fake_out_mem.ptr(),
            data.layout.dtype.size(out_size), ACL_MEMCPY_DEVICE_TO_DEVICE,
            handle->stream());

    // get out idx
    // collapse bool_mask
    TensorND collapse_bool_mask(
            bool_mask.raw_ptr(), bool_mask_layout.collapse_contiguous());
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
