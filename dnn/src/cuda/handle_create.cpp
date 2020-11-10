/**
 * \file dnn/src/cuda/handle_create.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"

#include "src/cuda/adaptive_pooling/opr_impl.h"
#include "src/cuda/add_update/opr_impl.h"
#include "src/cuda/argmxx/opr_impl.h"
#include "src/cuda/argsort/opr_impl.h"
#include "src/cuda/batch_normalization/opr_impl.h"
#include "src/cuda/batched_matrix_mul/opr_impl.h"
#include "src/cuda/checksum/opr_impl.h"
#include "src/cuda/concat/opr_impl.h"
#include "src/cuda/cond_take/opr_impl.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/convolution/opr_impl.h"
#include "src/cuda/convolution3d/opr_impl.h"
#include "src/cuda/convpooling/opr_impl.h"
#include "src/cuda/cumsum/opr_impl.h"
#include "src/cuda/cvt_color/opr_impl.h"
#include "src/cuda/dct/opr_impl.h"
#include "src/cuda/deformable_conv/opr_impl.h"
#include "src/cuda/deformable_ps_roi_pooling/opr_impl.h"
#include "src/cuda/dot/opr_impl.h"
#include "src/cuda/elemwise/opr_impl.h"
#include "src/cuda/elemwise_multi_type/opr_impl.h"
#include "src/cuda/eye/opr_impl.h"
#include "src/cuda/flip/opr_impl.h"
#include "src/cuda/gaussian_blur/opr_impl.h"
#include "src/cuda/group_local/opr_impl.h"
#include "src/cuda/images2neibs/opr_impl.h"
#include "src/cuda/indexing_multi_axis_vec/opr_impl.h"
#include "src/cuda/indexing_one_hot/opr_impl.h"
#include "src/cuda/linspace/opr_impl.h"
#include "src/cuda/local/opr_impl.h"
#include "src/cuda/local_share/opr_impl.h"
#include "src/cuda/lrn/opr_impl.h"
#include "src/cuda/mask_conv/opr_impl.h"
#include "src/cuda/matrix_inverse/opr_impl.h"
#include "src/cuda/matrix_mul/opr_impl.h"
#include "src/cuda/max_tensor_diff/opr_impl.h"
#include "src/cuda/mesh_indexing/opr_impl.h"
#include "src/cuda/param_pack/opr_impl.h"
#include "src/cuda/pooling/opr_impl.h"
#include "src/cuda/powc/opr_impl.h"
#include "src/cuda/reduce/opr_impl.h"
#include "src/cuda/relayout/opr_impl.h"
#include "src/cuda/relayout_format/opr_impl.h"
#include "src/cuda/repeat/opr_impl.h"
#include "src/cuda/resize/opr_impl.h"
#include "src/cuda/rng/opr_impl.h"
#include "src/cuda/roi_copy/opr_impl.h"
#include "src/cuda/roi_pooling/opr_impl.h"
#include "src/cuda/rotate/opr_impl.h"
#include "src/cuda/separable_conv/opr_impl.h"
#include "src/cuda/separable_filter/opr_impl.h"
#include "src/cuda/sleep/opr_impl.h"
#include "src/cuda/split/opr_impl.h"
#include "src/cuda/svd/opr_impl.h"
#include "src/cuda/tensor_remap/opr_impl.h"
#include "src/cuda/tile/opr_impl.h"
#include "src/cuda/topk/opr_impl.h"
#include "src/cuda/transpose/opr_impl.h"
#include "src/cuda/type_cvt/opr_impl.h"
#include "src/cuda/warp_affine/opr_impl.h"
#include "src/cuda/warp_perspective/opr_impl.h"
#include "src/cuda/winograd_filter_preprocess/opr_impl.h"
#include "src/cuda/local_share/opr_impl.h"
#include "src/cuda/roi_align/opr_impl.h"
#include "src/cuda/batch_conv_bias/opr_impl.h"
#include "src/cuda/remap/opr_impl.h"
#include "src/cuda/fake_quant/opr_impl.h"

namespace megdnn {
namespace cuda {

MEGDNN_FOREACH_OPR_CLASS(MEGDNN_SPECIALIZE_CREATE_OPERATOR)

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
