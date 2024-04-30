#include "opr_impl.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_fill_scalar.h"
#include "aclnnop/aclnn_s_where.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

#include <iostream>
#include <string>
#include <string_view>

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

void WhereForwardImpl::exec(
        _megdnn_tensor_in mask, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(mask.layout, data1.layout, data2.layout, dst.layout, workspace.size);
    AclTensor acl_mask(mask), acl_data1(data1), acl_data2(data2);
    AclTensor acl_dst(dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnSWhereGetWorkspaceSize(
            acl_mask.get(), acl_data1.get(), acl_data2.get(), acl_dst.get(), &ws_size,
            &executor));
    auto handle = concrete_handle(this->handle());
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnSWhere(ws.ptr(), ws_size, executor, handle->stream()));
}

void WhereBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in mask, _megdnn_tensor_out grad_data1,
        _megdnn_tensor_out grad_data2, _megdnn_workspace workspace) {
    check_exec(
            diff.layout, mask.layout, grad_data1.layout, grad_data2.layout,
            workspace.size);
    AclTensor acl_diff(diff), acl_mask(mask), acl_grad_data1(grad_data1),
            acl_grad_data2(grad_data2);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    auto handle = concrete_handle(this->handle());

    TensorLayout zero_layout(diff.layout, dtype::Int32());
    AclMem zero_mem(zero_layout.access_bytes(), handle);
    AclTensor zero_tensor(zero_mem.ptr(), zero_layout);
    AclScalar zero_scalar(0);
    aclnn_check(aclnnInplaceFillScalarGetWorkspaceSize(
            zero_tensor.get(), zero_scalar.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceFillScalar(ws.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnSWhereGetWorkspaceSize(
            acl_mask.get(), acl_diff.get(), zero_tensor.get(), acl_grad_data1.get(),
            &ws_size, &executor));
    AclMem ws2(ws_size, handle);
    aclnn_check(aclnnSWhere(ws2.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnSWhereGetWorkspaceSize(
            acl_mask.get(), zero_tensor.get(), acl_diff.get(), acl_grad_data2.get(),
            &ws_size, &executor));
    AclMem ws3(ws_size, handle);
    aclnn_check(aclnnSWhere(ws3.ptr(), ws_size, executor, handle->stream()));
}