#pragma once

#include "megdnn/handle.h"
#include "src/atlas/handle.h"

#include <acl/acl_base.h>

#define acl_check(_x)                                        \
    do {                                                     \
        aclError _ret = (_x);                                \
        if (_ret != ACL_ERROR_NONE) {                        \
            ::megdnn::atlas::__throw_acl_error__(_ret, #_x); \
        }                                                    \
    } while (0)

#define aclnn_check(_x)                                      \
    do {                                                     \
        aclnnStatus _ret = (_x);                             \
        if (_ret != OK) {                                    \
            ::megdnn::atlas::__throw_acl_error__(_ret, #_x); \
        }                                                    \
    } while (0)

#define aclnn_call(handle, AclnnOp, ...)                                           \
    do {                                                                           \
        uint64_t AclnnOp##_ws_size = 0;                                            \
        aclOpExecutor* AclnnOp##_executor = nullptr;                               \
        aclnn_check(AclnnOp##GetWorkspaceSize(                                     \
                __VA_ARGS__, &AclnnOp##_ws_size, &AclnnOp##_executor));            \
        AclMem AclnnOp##_ws(AclnnOp##_ws_size, handle);                            \
        aclnn_check(                                                               \
                AclnnOp(AclnnOp##_ws.ptr(), AclnnOp##_ws_size, AclnnOp##_executor, \
                        handle->stream()));                                        \
    } while (0)

#define AclTempTensor(handle, var, var_layout)           \
    AclMem var##_buf(var_layout.access_bytes(), handle); \
    AclTensor var(var##_buf.ptr(), var_layout)

// aclrtMemcpyAsync requests the addr of src and dst is 64 bytes align.
#define acl_safe_memcpy_async(dst, size0, src, size1, copy_type, stream)            \
    do {                                                                            \
        if (reinterpret_cast<uintptr_t>(dst) % 64 != 0 ||                           \
            reinterpret_cast<uintptr_t>(src) % 64 != 0) {                           \
            acl_check(aclrtSynchronizeStream(stream));                              \
            acl_check(aclrtMemcpy(dst, size0, src, size1, copy_type));              \
        } else {                                                                    \
            acl_check(aclrtMemcpyAsync(dst, size0, src, size1, copy_type, stream)); \
        }                                                                           \
    } while (0)

#define acl_safe_memcpy_async_with_sync(dst, size0, src, size1, copy_type, stream)  \
    do {                                                                            \
        if (reinterpret_cast<uintptr_t>(dst) % 64 != 0 ||                           \
            reinterpret_cast<uintptr_t>(src) % 64 != 0) {                           \
            acl_check(aclrtSynchronizeStream(stream));                              \
            acl_check(aclrtMemcpy(dst, size0, src, size1, copy_type));              \
        } else {                                                                    \
            acl_check(aclrtMemcpyAsync(dst, size0, src, size1, copy_type, stream)); \
            acl_check(aclrtSynchronizeStream(stream));                              \
        }                                                                           \
    } while (0)

namespace megdnn {
namespace atlas {

inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<atlas::HandleImpl*>(handle);
}

//! Error handling funcions
MEGDNN_NORETURN void __throw_acl_error__(aclError err, const char* msg);

template <typename T>
void print_tensor(const void* ptr, const TensorLayout& lyt) {
    lyt.dtype.assert_is_ctype<T>();
    printf("%s\n", lyt.to_string().c_str());
    size_t elem_bytes = lyt.access_bytes();
    SmallVector<uint8_t> host(elem_bytes);

    acl_check(aclrtSynchronizeDevice());
    acl_check(aclrtMemcpy(
            host.data(), elem_bytes, ptr, elem_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    acl_check(aclrtSynchronizeDevice());

    T* data = reinterpret_cast<T*>(host.data());
    for (size_t i = 0; i < elem_bytes / sizeof(T); ++i) {
        printf("%f ", float(data[i]));
    }
    printf("\n");
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
