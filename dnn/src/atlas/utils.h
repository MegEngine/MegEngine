/**
 * \file dnn/src/atlas/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
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
