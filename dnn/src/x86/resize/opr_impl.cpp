/**
 * \file dnn/src/x86/resize/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/resize/opr_impl.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/x86/handle.h"

#include <cstring>
#include "src/common/utils.h"

#include <pmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

#include "src/x86/handle.h"
#include "src/x86/resize/opr_impl.h"
#include "src/x86/resize/resize_cv.h"
#include "src/x86/utils.h"

using namespace megdnn;
using namespace x86;

void ResizeImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    if (param().format == param::Resize::Format::NCHW ||
        (src.layout[3] != 1 && src.layout[3] != 3) || !is_supported(SIMDType::SSE4_2) ||
        !is_nhwc_contig_wc(src.layout)) {
        fallback::ResizeImpl::exec(src, dst, workspace);
    } else {
        megdnn_assert(
                param().format == param::Resize::Format::NHWC, "invalid resize format");
        MEGDNN_DISPATCH_CPU_KERN_OPR(resize_cv_exec(src, dst, param().imode));
    }
}

// vim: syntax=cpp.doxygen
