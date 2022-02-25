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

namespace megdnn {
namespace atlas {

inline HandleImpl* concrete_handle(Handle* handle) {
    return static_cast<atlas::HandleImpl*>(handle);
}

//! Error handling funcions
MEGDNN_NORETURN void __throw_acl_error__(aclError err, const char* msg);

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
