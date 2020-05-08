/**
 * \file dnn/src/arm_common/warp_affine/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/handle.h"
#include "src/arm_common/warp_affine/opr_impl.h"
#include "src/arm_common/warp_affine/warp_affine_cv.h"
#include "src/common/warp_common.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_warpaffine)

using namespace megdnn;
using namespace arm_common;
using namespace megcv;

void WarpAffineImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                          _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, mat.layout, dst.layout, workspace.size);
    if (warp::is_cv_available(src.layout, mat.layout, dst.layout, param().imode,
                              param().format)) {
        MIDOUT_BEGIN(megdnn_arm_warpaffine, void) {
            warp_affine_cv_exec(src, mat, dst, param().border_val,
                                param().border_mode, param().imode, handle());
        }
        MIDOUT_END();
    } else {
        //! Use fallback implementation
        naive::WarpAffineImpl::exec(src, mat, dst, workspace);
    }
}

// vim: syntax=cpp.doxygen
