/**
 * \file dnn/src/naive/local_share/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/local_share/opr_impl.h"
#include "src/naive/convolution/helper.h"

#include <cstring>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
using namespace convolution;

namespace {

template <typename stype, typename ftype, typename dtype, typename comp_type,
          class Strategy>
void naive_kern(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                _megdnn_tensor_out dst, LocalShare::Param param) {
    size_t spatial_start, channel_pos, kern_spatial_start;
    spatial_start = 2;
    channel_pos = 1;
    kern_spatial_start = 3;
    size_t groups = 1;
    if (param.sparse == LocalShare::Param::Sparse::GROUP) {
        kern_spatial_start = 4;
        groups = filter.layout.shape[0];
    }

    auto N = src.layout.shape[0], IC = src.layout.shape[channel_pos],
         IH = src.layout.shape[spatial_start],
         IW = src.layout.shape[spatial_start + 1];
    auto FH = filter.layout.shape[kern_spatial_start],
         FW = filter.layout.shape[kern_spatial_start + 1];
    auto OC = dst.layout.shape[channel_pos],
         OH = dst.layout.shape[spatial_start],
         OW = dst.layout.shape[spatial_start + 1];
    size_t icpg = IC / groups, ocpg = OC / groups;

    size_t SGH = param.spatial_groups_h, SGW = param.spatial_groups_w;
    size_t GRP_OH = OH / SGH, GRP_OW = OW / SGW;

    size_t FS_G, FS_OC, FS_IC, FS_SPATIAL;
    // sgh, sgw, ic, fh, fw, oc
    FS_OC = 1;
    FS_SPATIAL = FS_OC * ocpg;
    FS_IC = FH * FW * FS_SPATIAL;
    FS_G = FS_IC * icpg * SGH * SGW;

    size_t PH = param.pad_h, PW = param.pad_w;
    size_t SH = param.stride_h, SW = param.stride_w;
    size_t dh = param.dilate_h, dw = param.dilate_w;
    megdnn_assert(param.dilate_h == 1 && param.dilate_w == 1);
    stype* __restrict sptr = src.compatible_ptr<stype>();
    ftype* __restrict fptr = filter.compatible_ptr<ftype>();
    dtype* __restrict dptr = dst.compatible_ptr<dtype>();

    int h_offset = -PH, w_offset = -PW;

    auto get_linear_addr = [](ptrdiff_t n, ptrdiff_t c, ptrdiff_t h,
                              ptrdiff_t w,
                              const TensorLayout& layout) -> ptrdiff_t {
        return n * layout.stride[0] + c * layout.stride[1] +
               h * layout.stride[2] + w * layout.stride[3];
    };

    auto get_filter_addr = [&](GroupCounter& gc_out, size_t ic, size_t ic0,
                               size_t fh, size_t fw) {
        return gc_out.cur_grp * FS_G + gc_out.cur_off * FS_OC +
               (ic - ic0) * FS_IC + (fh * FW + fw) * FS_SPATIAL;
    };

    for (size_t n = 0; n < N; ++n) {
        GroupCounter gc_out{ocpg};
        for (size_t oc = 0; oc < OC; ++oc, gc_out.next()) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    comp_type dval =
                            dptr[get_linear_addr(n, oc, oh, ow, dst.layout)];
                    Strategy::init_dval(dval);
                    size_t grp_oh = oh / GRP_OH, grp_ow = ow / GRP_OW;
                    ftype* fptr_cur = fptr + (grp_oh * SGW + grp_ow) * ocpg *
                                                     icpg * FH * FW;

                    for (size_t fh = 0; fh < FH; ++fh) {
                        for (size_t fw = 0; fw < FW; ++fw) {
                            uint32_t ih = SH * oh + fh * dh + h_offset,
                                     iw = SW * ow + fw * dw + w_offset;
                            // here ih and iw are represented in unsigned int
                            // they will become very large if underflow occurs
                            if (ih < IH && iw < IW) {
                                size_t ic0 = gc_out.cur_grp * icpg,
                                       ic1 = ic0 + icpg;
                                for (size_t ic = ic0; ic < ic1; ++ic) {
                                    stype& sval = sptr[get_linear_addr(
                                            n, ic, ih, iw, src.layout)];
                                    ftype& fval = fptr_cur[get_filter_addr(
                                            gc_out, ic, ic0, fh, fw)];
                                    Strategy::on(sval, fval, dval,
                                                 src.layout.dtype,
                                                 filter.layout.dtype,
                                                 dst.layout.dtype);
                                }
                            }
                        }
                    }
                    Strategy::write(
                            dval,
                            dptr[get_linear_addr(n, oc, oh, ow, dst.layout)]);
                }
            }
        }
    }
}
}  // namespace

void LocalShareForwardImpl::exec(_megdnn_tensor_in src,
                                 _megdnn_tensor_in filter,
                                 _megdnn_tensor_out dst,
                                 _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            (naive_kern<dt_float32, dt_float32, dt_float32, dt_float32,
                        StrategyFwd>(src, filter, dst, param())););
}

void LocalShareBackwardDataImpl::exec(_megdnn_tensor_in filter,
                                      _megdnn_tensor_in diff,
                                      _megdnn_tensor_out grad,
                                      _megdnn_workspace workspace) {
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            (naive_kern<dt_float32, dt_float32, dt_float32, dt_float32,
                        StrategyBwdData>(grad, filter, diff, param())););
}

void LocalShareBackwardFilterImpl::exec(_megdnn_tensor_in src,
                                        _megdnn_tensor_in diff,
                                        _megdnn_tensor_out grad,
                                        _megdnn_workspace workspace) {
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            (naive_kern<dt_float32, dt_float32, dt_float32, dt_float32,
                        StrategyBwdFlt>(src, grad, diff, param())););
}

std::vector<LocalShareForward::Algorithm*>
LocalShareForwardImpl::get_all_algorithms(const TensorLayout&,
                                          const TensorLayout&,
                                          const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())->default_local_share_fwd_algo()};
}

LocalShareForward::Algorithm* LocalShareForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_local_share_fwd_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

std::vector<LocalShareBackwardData::Algorithm*>
LocalShareBackwardDataImpl::get_all_algorithms(const TensorLayout&,
                                               const TensorLayout&,
                                               const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())
                    ->default_local_share_bwd_data_algo()};
}

LocalShareBackwardData::Algorithm*
LocalShareBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& /* filter */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo = static_cast<HandleImpl*>(handle())
                        ->default_local_share_bwd_data_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

std::vector<LocalShareBackwardFilter::Algorithm*>
LocalShareBackwardFilterImpl::get_all_algorithms(const TensorLayout&,
                                                 const TensorLayout&,
                                                 const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())
                    ->default_local_share_bwd_filter_algo()};
}

LocalShareBackwardFilter::Algorithm*
LocalShareBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo = static_cast<HandleImpl*>(handle())
                        ->default_local_share_bwd_filter_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

// vim: syntax=cpp.doxygen
