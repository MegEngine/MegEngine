#pragma once

#include "src/common/utils.h"
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {

bool chanwise_avx2_stride1_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride1_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride1_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

bool chanwise_avx2_stride2_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride2_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool chanwise_avx2_stride2_qint8_usable_preferred(
        const ConvBiasImpl::NCBKernSizeParam&);

bool direct_avx2_stride1_int8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride1_int8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride1_int8_usable_preferred(const ConvBiasImpl::NCBKernSizeParam&);

bool direct_avx2_stride2_int8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride2_int8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool direct_avx2_stride2_int8_usable_preferred(const ConvBiasImpl::NCBKernSizeParam&);

#if MEGDNN_X86_WITH_MKL_DNN
bool mkldnn_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_qint8_usable_preferred(const ConvBiasImpl::NCBKernSizeParam&);

bool mkldnn_matmul_qint8_usable(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_matmul_qint8_preferred(const ConvBiasImpl::NCBKernSizeParam&);
bool mkldnn_matmul_qint8_usable_preferred(const ConvBiasImpl::NCBKernSizeParam&);
#endif

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
