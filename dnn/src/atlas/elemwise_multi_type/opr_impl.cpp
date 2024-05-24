#pragma once

#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_eq_tensor.h"
#include "aclnnop/aclnn_equal.h"
#include "aclnnop/aclnn_lt_tensor.h"
#include "aclnnop/aclnn_ne_tensor.h"
#include "megdnn/tensor_iter.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"
#include "src/atlas/utils.h"
#include "src/common/elemwise_multi_type/opr_impl_helper.h"

using namespace megdnn;
using namespace atlas;

void print_shape(const AclTensor& tensor) {
    int64_t shapes[10];
    int64_t* shape_ptr = &shapes[0];
    uint64_t ndim;
    aclGetViewShape(tensor.get(), &shape_ptr, &ndim);
    for (int i = 0; i < ndim; i++) {
        printf("%d\t", shapes[i]);
    }
    printf("\n");
}

template <typename T>
void print(T t) {
    if (typeid(T) == typeid(double) || typeid(T) == typeid(float)) {
        printf("%f", t);
    } else if (typeid(T) == typeid(int)) {
        printf("%d", t);
    } else if (typeid(T) == typeid(bool)) {
        printf(t ? "true" : "false");
    }
}

template <typename T>
void print(std::string name, void* ptr, const TensorShape& shape, HandleImpl* handle) {
    auto size = shape.total_nr_elems();
    std::vector<T> resultData(size, 0);
    aclrtMemcpyAsync(
            resultData.data(), size * sizeof(T), ptr, size * sizeof(T),
            ACL_MEMCPY_DEVICE_TO_HOST, handle->stream());
    aclrtSynchronizeStream(handle->stream());
    printf("%s\n", name.c_str());
    for (int64_t idx = 0; idx < size; idx++) {
        print<T>(resultData[idx]);
        printf(",\t");
    }
    printf("\n");
}

void print_bool(
        std::string name, void* ptr, const TensorShape& shape, HandleImpl* handle) {
    auto size = shape.total_nr_elems();
    bool arr[size];
    aclrtMemcpyAsync(
            arr, size * sizeof(bool), ptr, size * sizeof(bool),
            ACL_MEMCPY_DEVICE_TO_HOST, handle->stream());
    aclrtSynchronizeStream(handle->stream());
    printf("%s\n", name.c_str());
    for (int64_t idx = 0; idx < size; idx++) {
        print<bool>(arr[idx]);
        printf(",\t");
    }
    printf("\n");
}

void ElemwiseMultiTypeImpl::dest_type_bool_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    megdnn_assert(
            param[0].layout.total_nr_elems() == param[1].layout.total_nr_elems() &&
            param[1].layout.total_nr_elems() == dst.layout.total_nr_elems());

    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    TensorLayout formated_shape = param[0].layout;
    for (int i = 0; i < 2; i++) {
        if (!param[i].layout.is_contiguous()) {
            formated_shape = param[i].layout;
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        const TensorND& inp = param[i];
        TensorLayout inp_layout = inp.layout;
        if (!inp_layout.eq_shape(formated_shape)) {
            inp_layout = inp_layout.reshape(formated_shape);
        }
        TensorND formated_inp(inp.raw_ptr(), inp_layout);
        acl_inps.push_back(formated_inp);
    }

    TensorLayout dst_layout = dst.layout;
    if (!dst_layout.eq_shape(formated_shape)) {
        dst_layout = dst_layout.reshape(formated_shape);
    }
    TensorND format_dst(dst.raw_ptr(), dst_layout);
    AclTensor acl_out(format_dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    if (mode == Elemwise::Mode::EQ) {
        aclnn_check(aclnnEqTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnEqTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::LEQ) {
        // x== y
        AclMem out_1_mem(format_dst.layout.access_bytes(), handle);
        AclTensor out_1_tensor(out_1_mem.ptr(), format_dst.layout);

        aclnn_check(aclnnEqTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), out_1_tensor.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnEqTensor(ws.ptr(), ws_size, executor, handle->stream()));

        AclMem out_2_mem(format_dst.layout.access_bytes(), handle);
        AclTensor out_2_tensor(out_2_mem.ptr(), format_dst.layout);

        // x < y
        aclnn_check(aclnnLtTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), out_2_tensor.get(), &ws_size,
                &executor));
        AclMem ws_2(ws_size, handle);
        aclnn_check(aclnnLtTensor(ws_2.ptr(), ws_size, executor, handle->stream()));

        // x <=y  is  (x == y) or (x < y)
        aclnn_check(aclnnBitwiseOrTensorGetWorkspaceSize(
                out_1_tensor.get(), out_2_tensor.get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws_3(ws_size, handle);
        aclnn_check(
                aclnnBitwiseOrTensor(ws_3.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::LT) {
        aclnn_check(aclnnLtTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLtTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::NEQ) {
        aclnn_check(aclnnNeTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnNeTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else {
        megdnn_assert_internal(0);
    }
}