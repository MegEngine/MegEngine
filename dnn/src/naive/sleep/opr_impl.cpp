/**
 * \file dnn/src/naive/sleep/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"

#if __DEPLOY_ON_XP_SP2__
#define MEGDNN_NO_THREAD 1
#endif

#include "src/naive/handle.h"
#if !MEGDNN_NO_THREAD
#include <thread>
#endif

namespace megdnn {
namespace naive {

void SleepForwardImpl::exec() {
#if MEGDNN_NO_THREAD
    megdnn_trap();
#else
    double seconds = m_param.time;
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<uint64_t>(seconds * 1e6))););
#endif
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
