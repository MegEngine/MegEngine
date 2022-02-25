#pragma once

#include "megdnn/basic_types.h"
#include "midout.h"
#include "src/common/conv_bias.h"
#include "src/common/opr_delegate.h"
#include "src/common/postprocess.h"

namespace {
#define POST_PROCESS_UNUSED_VAR()       \
    MEGDNN_MARK_USED_VAR(conv_dst_ptr); \
    MEGDNN_MARK_USED_VAR(bias_ptr);     \
    MEGDNN_MARK_USED_VAR(dst_ptr);      \
    MEGDNN_MARK_USED_VAR(bias_mode);    \
    MEGDNN_MARK_USED_VAR(nonlineMode);  \
    MEGDNN_MARK_USED_VAR(bias_type);    \
    MEGDNN_MARK_USED_VAR(dst_type);     \
    MEGDNN_MARK_USED_VAR(N);            \
    MEGDNN_MARK_USED_VAR(OC);           \
    MEGDNN_MARK_USED_VAR(OH);           \
    MEGDNN_MARK_USED_VAR(OW);           \
    MEGDNN_MARK_USED_VAR(pack_oc_size)

void to_handle_bias_and_nonlinear(
        void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
        megdnn::ConvBiasForward::BiasMode bias_mode,
        megdnn::param::ConvBias::NonlineMode nonlineMode, megdnn::DType bias_type,
        megdnn::DType dst_type, size_t N, size_t OC, size_t OH, size_t OW) {
    auto handle = megdnn::inplace_cpu_handle();
    auto conv_dst_tensor_layout = megdnn::TensorLayout({N, OC, OH, OW}, dst_type);
    auto conv_dst_tensor = megdnn::TensorND{conv_dst_ptr, conv_dst_tensor_layout};
    auto dst_tensor = megdnn::TensorND{dst_ptr, conv_dst_tensor_layout};
    auto bias_tensor_layout = conv_dst_tensor_layout;
    if (megdnn::ConvBiasForward::BiasMode::BROADCAST_CHANNEL_BIAS == bias_mode) {
        bias_tensor_layout = megdnn::TensorLayout({1, OC, 1, 1}, bias_type);
    } else if (megdnn::ConvBiasForward::BiasMode::NO_BIAS == bias_mode) {
        bias_tensor_layout = megdnn::TensorLayout({}, bias_type);
    }
    auto bias_tensor =
            megdnn::TensorND{const_cast<void*>(bias_ptr), bias_tensor_layout};
    handle_bias_and_nonlinear(
            handle.get(), nonlineMode, &conv_dst_tensor, &dst_tensor, &bias_tensor);
}

template <
        typename ctype, typename dtype = ctype,
        megdnn::PostprocessMode postprocess_mode = megdnn::PostprocessMode::FLOAT>
struct PostProcess {
    static void run(
            void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
            megdnn::ConvBiasForward::BiasMode bias_mode,
            megdnn::param::ConvBias::NonlineMode nonlineMode, megdnn::DType bias_type,
            megdnn::DType dst_type, size_t N, size_t OC, size_t OH, size_t OW,
            size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        to_handle_bias_and_nonlinear(
                conv_dst_ptr, bias_ptr, dst_ptr, bias_mode, nonlineMode, bias_type,
                dst_type, N, OC, OH, OW);
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::NO_PROCESS> {
    static void run(
            void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
            megdnn::ConvBiasForward::BiasMode bias_mode,
            megdnn::param::ConvBias::NonlineMode nonlineMode, megdnn::DType bias_type,
            megdnn::DType dst_type, size_t N, size_t OC, size_t OH, size_t OW,
            size_t pack_oc_size = 1) {
        POST_PROCESS_UNUSED_VAR();
    }
};

template <typename opctype, typename opdtype>
struct PostProcess<opctype, opdtype, megdnn::PostprocessMode::QUANTIZED> {
    static void run(
            void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
            megdnn::ConvBiasForward::BiasMode bias_mode,
            megdnn::param::ConvBias::NonlineMode nonlineMode, megdnn::DType bias_type,
            megdnn::DType dst_type, size_t N, size_t OC, size_t OH, size_t OW,
            size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        to_handle_bias_and_nonlinear(
                conv_dst_ptr, bias_ptr, dst_ptr, bias_mode, nonlineMode, bias_type,
                dst_type, N, OC, OH, OW);
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::ADD_BIAS> {
    static void run(
            void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
            megdnn::ConvBiasForward::BiasMode bias_mode,
            megdnn::param::ConvBias::NonlineMode nonlineMode, megdnn::DType bias_type,
            megdnn::DType dst_type, size_t N, size_t OC, size_t OH, size_t OW,
            size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        if (bias_mode == megdnn::ConvBiasForward::BiasMode::NO_BIAS) {
            return;
        }
        to_handle_bias_and_nonlinear(
                conv_dst_ptr, bias_ptr, dst_ptr, bias_mode, nonlineMode, bias_type,
                dst_type, N, OC, OH, OW);
    }
};

}  // namespace
