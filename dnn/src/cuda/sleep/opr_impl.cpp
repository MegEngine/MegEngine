/**
 * \file dnn/src/cuda/sleep/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./kern.cuh"

#include "src/cuda/handle.h"

namespace megdnn {
namespace cuda {

void SleepForwardImpl::exec() {
    double seconds = m_param.time;
    megdnn_assert(seconds > 0);
    auto hdl = static_cast<HandleImpl*>(handle());
    sleep(hdl->stream(), hdl->device_prop().clockRate * 1e3 * seconds * 1.2);
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
