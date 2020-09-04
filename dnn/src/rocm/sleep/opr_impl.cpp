/**
 * \file dnn/src/rocm/sleep/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "./kern.h.hip"

#include "src/rocm/handle.h"

namespace megdnn {
namespace rocm {

void SleepForwardImpl::exec() {
    double seconds = m_param.time;
    megdnn_assert(seconds > 0);
    auto hdl = static_cast<HandleImpl*>(handle());
    sleep(hdl->stream(), hdl->device_prop().clockRate * 1000 * seconds);
}

} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen

