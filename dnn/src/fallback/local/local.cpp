/**
 * \file dnn/src/fallback/local/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/local/opr_impl.h"

#include "src/common/utils.h"
#include "src/fallback/handle.h"
#include <cstring>

using namespace megdnn;
using namespace fallback;
namespace {
    size_t get_type_bytes(DType type){
        if(type == dtype::Float32() || type == dtype::Int32()){
            return 4;
        }
        else if(type == dtype::Uint8() || type == dtype::Int8() || type == dtype::Byte()){
            return 1;
        }
        else if(type == dtype::Int16() || type == dtype::Float16() || type == dtype::BFloat16() || type == dtype::Uint16()){
            return 2;
        }
        else return 8;
    }
}

size_t LocalForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
                const TensorLayout &filter,
                const TensorLayout &dst){
  auto IC = src.shape[1];
  auto FH = filter.shape[3];
  auto FW = filter.shape[4];
  size_t type_bytes = std::max(get_type_bytes(src.dtype),get_type_bytes(filter.dtype));
  type_bytes = std::max(type_bytes,get_type_bytes(dst.dtype));
  return FH * FW * IC * 4 * type_bytes + 64;
}

LocalForwardImpl::float_noncontig_batch_kern
LocalForwardImpl::dispatch_float_noncontig_batch(
        const TensorLayout &src,
        const TensorLayout &/*filter*/,
        const TensorLayout &/*dst*/) {
    if (src.dtype == dtype::Float32()) {
        if (param().mode == Mode::CROSS_CORRELATION) {
            return &fallback_kern<true, float>;
        } else {
            return &fallback_kern<false, float>;
        }
    } else if (DNN_FLOAT16_SELECT(src.dtype == dtype::Float16(), false)) {
        DNN_INC_FLOAT16(
        megdnn_assert(src.dtype == dtype::Float16());
        if (param().mode == Mode::CROSS_CORRELATION) {
            return &fallback_kern<true MEGDNN_COMMA dt_float16>;
        } else {
            return &fallback_kern<false MEGDNN_COMMA dt_float16>;
        });
    } else {
        megdnn_assert_internal(false);
        return nullptr;
    }
}
#define UNROLL_CODE1(cb, a...) cb(0, ##a)
#define UNROLL_CODE2(cb, a...) cb(0, ##a) cb(1, ##a)
#define UNROLL_CODE3(cb, a...)                                                 \
  UNROLL_CODE2(cb, ##a)                                                        \
  cb(2, ##a)
#define UNROLL_CODE4(cb, a...)                                                 \
  UNROLL_CODE3(cb, ##a)                                                        \
  cb(3, ##a)
#define UNROLL_CODE(cb, i, a...) UNROLL_CODE##i(cb, ##a)

template<bool is_xcorr, typename dtype>
void LocalForwardImpl::fallback_kern(const FloatNoncontigBatchKernParam &param) {
    UNPACK_LOCAL_FLOAT_NONCONTIG_BATCH_KERN_PARAM(param, dtype);
    // for every batch
#define SET_H_W(i) auto h##i = (row_idx + i) / OW, w##i = (row_idx + i) % OW;
#define SET_PATCH_H(i) auto patch_Ah##i = h_offset + h##i * SH - PH;           \
    if(is_xcorr == false){                                                     \
        patch_Ah##i = FH - h_offset - 1 + h##i * SH - PH;                      \
    }
#define SET_PATCH_W(i) auto patch_Aw##i = w_offset + w##i * SW - PW;           \
    if(is_xcorr == false){                                                     \
        patch_Aw##i = FW - w_offset - 1 + w##i * SW - PW;                      \
    }
#define FILL_B(i)                                                              \
  if (patch_Ah##i >= IH || patch_Aw##i >= IW)                                  \
    B[col_idx + B_colNum * i] = 0;                                             \
  else                                                                         \
    B[col_idx + B_colNum * i] =                                                \
        A[c_offset * IH * IW + patch_Ah##i * IW + patch_Aw##i];
#define DELARE_SRC(i) auto src_row_ptr##i = B + i * B_colNum;
#define DELARE_FILTER(i)                                                       \
  auto filter_row_ptr##i = filter + (row_idx + i) * B_colNum * OC;
#define SET_DST(i)                                                             \
  dst_n[oc_idx * OH * OW + row_idx + i] +=                                     \
      src_row_ptr##i[elem_idx] * filter_row_ptr##i[elem_idx * OC + oc_idx];
#define EXEC(i)                                                                \
  for (; row_idx + i <= B_rowNum; row_idx += i) {                              \
    UNROLL_CODE(SET_H_W, i)                                                    \
    for (size_t col_idx = 0; col_idx < B_colNum; col_idx++) {                  \
      auto w_offset = col_idx % FW;                                            \
      auto h_offset = (col_idx / FW) % FH;                                     \
      auto c_offset = col_idx / (FH * FW);                                     \
      UNROLL_CODE(SET_PATCH_H, i)                                              \
      UNROLL_CODE(SET_PATCH_W, i)                                              \
      UNROLL_CODE(FILL_B, i)                                                   \
    }                                                                          \
    UNROLL_CODE(DELARE_SRC, i)                                                 \
    UNROLL_CODE(DELARE_FILTER, i)                                              \
    for (size_t oc_idx = 0; oc_idx < OC; oc_idx++) {                           \
      for (size_t elem_idx = 0; elem_idx < B_colNum; elem_idx++) {             \
        UNROLL_CODE(SET_DST, i)                                                \
      }                                                                        \
    }                                                                          \
  }
  auto B = workspace;
    for(size_t n = 0;n < N;n++){
        auto src_n = src + n * INP_BS;
        auto dst_n = dst + n * OUT_BS;
        auto B_colNum = IC * FH * FW;
        auto B_rowNum = OH * OW;
        auto A = src_n;
        memset(dst_n, 0, sizeof(dtype) * OUT_BS);
        size_t row_idx = 0;
        EXEC(4);
        EXEC(2);
        EXEC(1);
    }
#undef SET_H_W
#undef SET_PATCH_H
#undef SET_PATCH_W
#undef FILL_B
#undef DELARE_SRC
#undef DELARE_FILTER
#undef SET_DST
#undef EXEC
#undef UNROLL_CODE
#undef UNROLL_CODE1H
#undef UNROLL_CODE2
#undef UNROLL_CODE3
#undef UNROLL_CODE4
}

void LocalForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    exec_use_float_noncontig_batch(src, filter, dst, workspace);
}

LocalForwardImpl::FloatNoncontigBatchKernParam
LocalForwardImpl::make_float_kern_param(
        _megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) const {
    return {
        src.raw_ptr, filter.raw_ptr, dst.raw_ptr,
        // n
        src.layout.shape[0],
        // ic, ih, iw, oc, oh, ow, fh, fw
        src.layout.shape[1], src.layout.shape[2], src.layout.shape[3],
        dst.layout.shape[1], dst.layout.shape[2], dst.layout.shape[3],
        filter.layout.shape[3], filter.layout.shape[4],
        // ph, pw, sh, sw
        param().pad_h, param().pad_w, param().stride_h, param().stride_w,
        // inp_bs, out_bs
        src.layout.stride[0], dst.layout.stride[0],
        workspace.raw_ptr
    };
}

void LocalForwardImpl::exec_use_float_noncontig_batch(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {

    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    auto fp = make_float_kern_param(src, filter, dst, workspace);
    auto kptr = dispatch_float_noncontig_batch(
            src.layout, filter.layout, dst.layout);
    auto kern = [fp, kptr]() {
        kptr(fp);
    };
    static_cast<fallback::HandleImpl*>(handle())->dispatch_kern(kern);
}



// vim: syntax=cpp.doxygen
