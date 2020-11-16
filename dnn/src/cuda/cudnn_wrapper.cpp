/**
 * \file dnn/src/cuda/cudnn_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/cudnn_wrapper.h"

#include "src/common/utils.h"
#include "src/cuda/utils.h"

namespace {

using namespace megdnn;

cudnnDataType_t to_cudnn_dtype(DType type,
                               const param::Convolution::Format format = {}) {
    switch (type.enumv()) {
        case DTypeEnum::Float32:
            return CUDNN_DATA_FLOAT;
        case DTypeEnum::Float16:
            return CUDNN_DATA_HALF;
#if CUDNN_MAJOR >= 7
        case DTypeEnum::Int32:
        case DTypeEnum::QuantizedS32:
            return CUDNN_DATA_INT32;
#endif
#if CUDNN_MAJOR >= 6
        case DTypeEnum::QuantizedS8: {
            if (format == param::Convolution::Format::NCHW4)
                return CUDNN_DATA_INT8x4;
#if CUDNN_VERSION >= 7500
            else if (format == param::Convolution::Format::NCHW32)
                return CUDNN_DATA_INT8x32;
#endif
            else
                return CUDNN_DATA_INT8;
        }

        case DTypeEnum::Int8: {
            if (format == param::Convolution::Format::NCHW4)
                return CUDNN_DATA_INT8x4;
#if CUDNN_VERSION >= 7500
            else if (format == param::Convolution::Format::NCHW32)
                return CUDNN_DATA_INT8x32;
#endif
            else
                return CUDNN_DATA_INT8;
        }
#endif
        default:
#if CUDNN_MAJOR >= 6
    megdnn_throw(megdnn_mangle("dtype must be float16/float32/int8/int32"));
#else
    megdnn_throw(megdnn_mangle("dtype must be float16/float32"));
#endif
    }

}

cudnnTensorFormat_t to_cudnn_format(const param::Convolution::Format format) {
    switch (format) {
        case param::Convolution::Format::NCHW:
            return CUDNN_TENSOR_NCHW;
#if CUDNN_MAJOR >= 7
        case param::Convolution::Format::NCHW4:
        case param::Convolution::Format::NCHW32:
            return CUDNN_TENSOR_NCHW_VECT_C;
#endif
        case param::Convolution::Format::NHWC:
            return CUDNN_TENSOR_NHWC;
        default:
            megdnn_assert_internal(0);
    }
}

}  // namespace

namespace megdnn {
namespace cuda {

cudnnDataType_t get_compute_type_fp16(
        param::Convolution::ComputeMode comp_mode) {
    using Param = param::Convolution;
    cudnnDataType_t compute_type;
    if (comp_mode == Param::ComputeMode::DEFAULT) {
        // TRUE_HALF_CONFIG
        if (is_compute_capability_required(5, 3)) {
            compute_type = CUDNN_DATA_HALF;
        } else {
            auto&& device_prop = current_device_prop();
            int major = device_prop.major, minor = device_prop.minor;
            MEGDNN_MARK_USED_VAR(major);
            MEGDNN_MARK_USED_VAR(minor);
            megdnn_log_warn(
                    "TRUE_HALF_CONFIG only supported on architectures with "
                    "true fp16 support, i.e., compute capability 5.3 and "
                    "later (got %d.%d). Use PSEUDO_HALF_CONFIG instead",
                    major, minor);
            compute_type = CUDNN_DATA_FLOAT;
        }
    } else {
        megdnn_assert(comp_mode == Param::ComputeMode::FLOAT32);
        // PSEUDO_HALF_CONFIG
        compute_type = CUDNN_DATA_FLOAT;
    }
    return compute_type;
}

TensorDesc::TensorDesc() {
    cudnn_check(cudnnCreateTensorDescriptor(&desc));
}

TensorDesc::~TensorDesc() {
    cudnn_check(cudnnDestroyTensorDescriptor(desc));
}

void TensorDesc::set(const TensorLayout& layout,
                     const param::Convolution::Format format) {
    // Layout can be not contiguous; group conv needs it.
    // megdnn_assert_contiguous(layout);
    if (format == param::Convolution::Format::NCHW4 ||
        format == param::Convolution::Format::NCHW32)
        megdnn_assert_eq_size_t(layout.ndim, 5_z);
    else
        megdnn_assert_eq_size_t(layout.ndim, 4_z);

    size_t c_pos, spatial_pos;
    if (format == param::Convolution::Format::NCHW ||
        format == param::Convolution::Format::NCHW4 ||
        format == param::Convolution::Format::NCHW32) {
        c_pos = 1;
        spatial_pos = 2;
    } else {
        megdnn_assert(format == param::Convolution::Format::NHWC);
        c_pos = 3;
        spatial_pos = 1;
    }
    if (format == param::Convolution::Format::NCHW4) {
        megdnn_assert(layout.is_physical_contiguous());
        cudnn_check(cudnnSetTensor4dDescriptor(
                desc, to_cudnn_format(format),
                to_cudnn_dtype(layout.dtype, format), layout.shape[0],
                layout.shape[c_pos] * 4, layout.shape[spatial_pos + 0],
                layout.shape[spatial_pos + 1]));
    } else if (format == param::Convolution::Format::NCHW32) {
        megdnn_assert(layout.is_physical_contiguous());
        cudnn_check(cudnnSetTensor4dDescriptor(
                desc, to_cudnn_format(format),
                to_cudnn_dtype(layout.dtype, format), layout.shape[0],
                layout.shape[c_pos] * 32, layout.shape[spatial_pos + 0],
                layout.shape[spatial_pos + 1]));

    } else {
        cudnn_check(cudnnSetTensor4dDescriptorEx(
                desc, to_cudnn_dtype(layout.dtype), layout.shape[0],
                layout.shape[c_pos], layout.shape[spatial_pos + 0],
                layout.shape[spatial_pos + 1], layout.stride[0],
                layout.stride[c_pos], layout.stride[spatial_pos + 0],
                layout.stride[spatial_pos + 1]));
    }
}

template <typename Param>
FilterDesc<Param>::FilterDesc() {
    cudnn_check(cudnnCreateFilterDescriptor(&desc));
}

template <typename Param>
FilterDesc<Param>::~FilterDesc() {
    cudnn_check(cudnnDestroyFilterDescriptor(desc));
}

template <typename Param>
void FilterDesc<Param>::set(
        const typename ConvolutionBase<Param>::CanonizedFilterMeta&
                filter_meta) {
    megdnn_assert(filter_meta.spatial_ndim == 2);
#if CUDNN_VERSION < 7500
    megdnn_assert(filter_meta.dilation[0] == 1 && filter_meta.dilation[1] == 1);
#endif
#if CUDNN_MAJOR <= 6
    megdnn_assert(filter_meta.group == 1);
#endif

    auto filter_format = filter_meta.format;
    if (filter_format == param::ConvBias::Format::NCHW4_NCHW) {
        filter_format = param::ConvBias::Format::NCHW4;
    }
    // cuDNN version 6 or below filter_meta.group always is 1.
    // So it is compatible for all cuDNN versions.
    cudnn_check(cudnnSetFilter4dDescriptor(
            desc, to_cudnn_dtype(filter_meta.dtype, filter_format),
            to_cudnn_format(filter_format),
            filter_meta.ocpg * filter_meta.group,  // cudnn 6 group always be 1
            filter_meta.icpg, filter_meta.spatial[0], filter_meta.spatial[1]));
}

template class FilterDesc<param::Convolution>;
template class FilterDesc<param::ConvBias>;

ConvDesc::ConvDesc() {
    cudnn_check(cudnnCreateConvolutionDescriptor(&desc));
#if CUDNN_VERSION >= 7000
    // cudnn enables tensor core when tensors have dataType =
    // CUDNN_DATA_HALF, so it should be safe to enable globally
    cudnn_check(cudnnSetConvolutionMathType(desc, CUDNN_TENSOR_OP_MATH));
#endif
}

ConvDesc::~ConvDesc() {
    cudnn_check(cudnnDestroyConvolutionDescriptor(desc));
}

void ConvDesc::set(DType data_type, const param::Convolution& param,
                   const size_t nr_group) {
    using Param = param::Convolution;
    cudnnConvolutionMode_t mode;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw(megdnn_mangle("conv mode must be conv or xcorr."));
    }
    cudnnDataType_t compute_type;
    MEGDNN_MARK_USED_VAR(compute_type);
    if (data_type.enumv() == DTypeEnum::Float32) {
        // FLOAT_CONFIG
        compute_type = CUDNN_DATA_FLOAT;
    } else if (data_type.enumv() == DTypeEnum::Float16) {
        auto comp_mode = param.compute_mode;
        compute_type = get_compute_type_fp16(comp_mode);
#if CUDNN_MAJOR >= 7
    } else if (data_type.category() == DTypeCategory::INT ||
               data_type.category() == DTypeCategory::QUANTIZED) {
        compute_type = CUDNN_DATA_INT32;
#endif
    } else {
        megdnn_throw(megdnn_mangle("unspport data type for conv bias"));
    }
#if CUDNN_MAJOR >= 7
    cudnn_check(cudnnSetConvolutionGroupCount(desc, nr_group));
#else
    megdnn_assert(nr_group == 1);
#endif

#if CUDNN_MAJOR >= 6
    cudnn_check(cudnnSetConvolution2dDescriptor(
            desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode, compute_type));
#else
    cudnn_check(cudnnSetConvolution2dDescriptor(
            desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode));
#endif
}

PoolingDesc::PoolingDesc() {
    cudnn_check(cudnnCreatePoolingDescriptor(&desc));
}

PoolingDesc::~PoolingDesc() {
    cudnn_check(cudnnDestroyPoolingDescriptor(desc));
}

void PoolingDesc::set(const param::Pooling& param) {
    cudnnPoolingMode_t mode;
    switch (param.mode) {
        case param::Pooling::Mode::MAX:
            mode = CUDNN_POOLING_MAX;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            break;
    }
    cudnn_check(cudnnSetPooling2dDescriptor(
            desc, mode, CUDNN_NOT_PROPAGATE_NAN, param.window_h, param.window_w,
            param.pad_h, param.pad_w, param.stride_h, param.stride_w));
}

LRNDesc::LRNDesc() {
    cudnn_check(cudnnCreateLRNDescriptor(&desc));
}

LRNDesc::~LRNDesc() {
    cudnn_check(cudnnDestroyLRNDescriptor(desc));
}

void LRNDesc::set(const param::LRN& param) {
    megdnn_assert(param.n & 1, "n is %u", param.n);
    megdnn_assert(param.n >= CUDNN_LRN_MIN_N, "n is %u, CUDNN_LRN_MIN_N is %d",
                  param.n, CUDNN_LRN_MIN_N);
    megdnn_assert(param.n <= CUDNN_LRN_MAX_N, "n is %u, CUDNN_LRN_MAX_N is %d",
                  param.n, CUDNN_LRN_MAX_N);
    megdnn_assert(param.k >= CUDNN_LRN_MIN_K, "k is %f, CUDNN_LRN_MIN_K is %lf",
                  param.k, CUDNN_LRN_MIN_K);
    megdnn_assert(param.beta >= CUDNN_LRN_MIN_BETA,
                  "beta is %f, CUDNN_LRN_MIN_BETA is %lf", param.beta,
                  CUDNN_LRN_MIN_BETA);
    // Note that alpha is divided by n in the cudnn implementation,
    // so we have to multiply alpha by n ahead of time.
    cudnn_check(cudnnSetLRNDescriptor(desc, param.n, param.alpha * param.n,
                                      param.beta, param.k));
}

BNParamDesc::BNParamDesc() {
    cudnn_check(cudnnCreateTensorDescriptor(&desc));
}

void BNParamDesc::set(const cudnnTensorDescriptor_t xDesc,
                      cudnnBatchNormMode_t mode) {
    cudnn_check(cudnnDeriveBNTensorDescriptor(desc, xDesc, mode));
}

BNParamDesc::~BNParamDesc() {
    cudnn_check(cudnnDestroyTensorDescriptor(desc));
}

Tensor3DDesc::Tensor3DDesc() {
    cudnn_check(cudnnCreateTensorDescriptor(&desc));
}

Tensor3DDesc::~Tensor3DDesc() {
    cudnn_check(cudnnDestroyTensorDescriptor(desc));
}

int sc(const size_t x) {
    return static_cast<int>(x);
}
void Tensor3DDesc::set(const TensorLayout& layout, bool is_ndhwc) {
    megdnn_assert_eq_size_t(layout.ndim, 5_z);
    size_t c_pos, spatial_pos;
    if (is_ndhwc) {
        c_pos = 4;
        spatial_pos = 1;
    } else {  // ncdhw
        c_pos = 1;
        spatial_pos = 2;
    }
    const int dimA[] = {sc(layout.shape[0]), sc(layout.shape[c_pos]),
                        sc(layout.shape[spatial_pos + 0]),
                        sc(layout.shape[spatial_pos + 1]),
                        sc(layout.shape[spatial_pos + 2])};

    const int strideA[] = {sc(layout.stride[0]), sc(layout.stride[c_pos]),
                           sc(layout.stride[spatial_pos + 0]),
                           sc(layout.stride[spatial_pos + 1]),
                           sc(layout.stride[spatial_pos + 2])};

    cudnn_check(cudnnSetTensorNdDescriptor(desc, to_cudnn_dtype(layout.dtype),
                                           5, dimA, strideA));
}

Filter3DDesc::Filter3DDesc() {
    cudnn_check(cudnnCreateFilterDescriptor(&desc));
}

Filter3DDesc::~Filter3DDesc() {
    cudnn_check(cudnnDestroyFilterDescriptor(desc));
}

void Filter3DDesc::set(
        const Convolution3DBase::CanonizedFilterMeta& filter_meta) {
    megdnn_assert(filter_meta.spatial_ndim == 3);
#if CUDNN_MAJOR <= 6
    megdnn_assert(filter_meta.group == 1);
#endif

    // cuDNN version 6 or below filter_meta.group always is 1.
    // So it is compatible for all cuDNN versions.
    const int filterDimA[] = {
            sc(filter_meta.ocpg *
               filter_meta.group),  // cudnn 6 group always be 1
            sc(filter_meta.icpg), sc(filter_meta.spatial[0]),
            sc(filter_meta.spatial[1]), sc(filter_meta.spatial[2])};

    cudnn_check(cudnnSetFilterNdDescriptor(
            desc, to_cudnn_dtype(DType::from_enum(filter_meta.dtype_enum)),
            CUDNN_TENSOR_NCHW, 5, filterDimA));
}

Conv3DDesc::Conv3DDesc() {
    cudnn_check(cudnnCreateConvolutionDescriptor(&desc));

#if CUDNN_MAJOR >= 7
    // cudnn enables tensor core when tensors have dataType = CUDNN_DATA_HALF,
    // so it should be safe to enable globally
    cudnn_check(cudnnSetConvolutionMathType(desc, CUDNN_TENSOR_OP_MATH));
#endif
}

Conv3DDesc::~Conv3DDesc() {
    cudnn_check(cudnnDestroyConvolutionDescriptor(desc));
}

void Conv3DDesc::set(const param::Convolution3D& param, const size_t nr_group) {
    cudnnConvolutionMode_t mode;
    switch (param.mode) {
        case param::Convolution3D::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case param::Convolution3D::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw(megdnn_mangle("conv mode must be conv or xcorr."));
    }
#if CUDNN_MAJOR >= 7
    cudnn_check(cudnnSetConvolutionGroupCount(desc, nr_group));
#else
    megdnn_assert(nr_group == 1);
#endif

    const int padA[] = {sc(param.pad_d), sc(param.pad_h), sc(param.pad_w)},
              filterStrideA[] = {sc(param.stride_d), sc(param.stride_h),
                                 sc(param.stride_w)},
              dilationA[] = {sc(param.dilate_d), sc(param.dilate_h),
                             sc(param.dilate_w)};
    // not use true half
    // in CUDNN_MAJOR < 6, all elements in dilA shoule be 1
    cudnn_check(cudnnSetConvolutionNdDescriptor(
            desc, 3, padA, filterStrideA, dilationA, mode, CUDNN_DATA_FLOAT));
}

////////////////////////// CudnnAlgoPack //////////////////////////

#define V1(v) #v
#define V(v) V1(v)
#define DEF_NAME(NAME) \
    #NAME "v" V(CUDNN_MAJOR) "." V(CUDNN_MINOR) "." V(CUDNN_PATCHLEVEL)
#define DEF_ALGO(NAME, PROD)           \
    {                                  \
        NAME, { DEF_NAME(NAME), PROD } \
    }

#if !(CUDNN_MAJOR >= 6 || CUDNN_MINOR >= 1)
#pragma message "not latest cudnn"
#endif

const std::unordered_map<cudnnConvolutionBwdDataAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv_bwd_data_algos() {
    static const std::unordered_map<cudnnConvolutionBwdDataAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, false),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, true),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT, true),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, true),
#if CUDNN_MAJOR >= 5
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD, true),
#if CUDNN_MAJOR >= 6 || CUDNN_MINOR >= 1
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
                         true),
#endif
#endif
            };

    return algos;
}

const std::unordered_map<cudnnConvolutionBwdFilterAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv_bwd_flt_algos() {
    static const std::unordered_map<cudnnConvolutionBwdFilterAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, false),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, true),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT, true),
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, false),
#if CUDNN_MAJOR >= 6 || (CUDNN_MAJOR >= 5 && CUDNN_MINOR >= 1)
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
                         true),
#if CUDNN_MAJOR >= 6
                DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING, true),
#endif
#endif

            };

    return algos;
}


const std::unordered_map<cudnnConvolutionFwdAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv_fwd_algos() {
    static const std::unordered_map<cudnnConvolutionFwdAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, true),
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                         true),
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_GEMM, true),
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, true),
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_FFT, true),
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, true),

#if CUDNN_MAJOR >= 5
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, true),
#if CUDNN_MAJOR >= 6 || CUDNN_MINOR >= 1
                DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, true),
#endif
#endif

            };

    return algos;
}

const std::unordered_map<cudnnConvolutionBwdDataAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv3d_bwd_data_algos() {
    static const std::unordered_map<cudnnConvolutionBwdDataAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, false),
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, true),
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING, true),
            };

    return algos;
}  // namespace cuda

const std::unordered_map<cudnnConvolutionBwdFilterAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv3d_bwd_flt_algos() {
#pragma message \
        "fp16 dilated conv with odd size filter, only algo_1 works, need focus on doc"
    static const std::unordered_map<cudnnConvolutionBwdFilterAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, false),
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, true),
                    DEF_ALGO(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3, false),
            };

    return algos;
}

const std::unordered_map<cudnnConvolutionFwdAlgo_t, CudnnAlgoPack::Attr>
CudnnAlgoPack::conv3d_fwd_algos() {
    static const std::unordered_map<cudnnConvolutionFwdAlgo_t,
                                    CudnnAlgoPack::Attr>
            algos = {
                    DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, true),
                    DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                             true),
                    DEF_ALGO(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, true),
            };

    return algos;
}

#undef DEF_ALGO
#undef DEF_NAME
#undef V
#undef V1

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
