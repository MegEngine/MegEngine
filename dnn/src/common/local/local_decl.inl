/**
 * \file dnn/src/common/local/local_decl.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// simd_macro/*_helper.h should be included before including this file.
//
// The following functions would be declared in this file:
//
// void local_xcorr_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
// void local_conv_MEGDNN_SIMD_NAME(const LocalKParam &kparam);
//
#include "src/naive/local/opr_impl.h"

#include "src/common/macro_helper.h"

namespace megdnn {

using LocalKParam = naive::LocalForwardImpl::FloatNoncontigBatchKernParam;

void WITH_SIMD_SUFFIX(local_xcorr)(
        const LocalKParam &param) MEGDNN_SIMD_ATTRIBUTE_TARGET;

void WITH_SIMD_SUFFIX(local_conv)(
        const LocalKParam &param) MEGDNN_SIMD_ATTRIBUTE_TARGET;

} // namespace megdnn

#include "src/common/macro_helper_epilogue.h"
