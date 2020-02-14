/**
 * \file dnn/src/fallback/roi_copy/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/roi_copy/opr_impl.h"
#include "src/fallback/handle.h"

#include "src/common/cv/common.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace fallback {

void ROICopyImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                       _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    size_t N = dst.layout.shape[0], OH = dst.layout.shape[1],
           OW = dst.layout.shape[2], OC = dst.layout.shape[3];
    ptrdiff_t istride0 = src.layout.stride[0], istride1 = src.layout.stride[1],
              istride2 = src.layout.stride[2], istride3 = src.layout.stride[3];

    TensorLayout relayout_src_layout({N, OH, OW, OC},
                                     {istride0, istride1, istride2, istride3},
                                     src.layout.dtype);
    TensorND relayout_src(
        static_cast<char*>(src.raw_ptr) +
            (param().row_from * istride1 + param().col_from * istride2) *
                src.layout.dtype.size(),
        relayout_src_layout);
    static_cast<HandleImplHelper*>(handle())->relayout_opr()->exec(
        relayout_src, dst);
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
