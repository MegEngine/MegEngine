/**
 * \file dnn/src/cuda/api_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/common/api_cache.h"
#include "src/cuda/cudnn_wrapper.h"

namespace megdnn {
    class CudnnConvDescParam {
    public:
        cudnnConvolutionDescriptor_t value;
        Empty serialize(StringSerializer& ser, Empty) {
            int ndim = MEGDNN_MAX_NDIM;
            int padA[MEGDNN_MAX_NDIM];
            int strideA[MEGDNN_MAX_NDIM];
            int dilationA[MEGDNN_MAX_NDIM];
            cudnnConvolutionMode_t mode;
            cudnnDataType_t computeType;
            cudnnGetConvolutionNdDescriptor(value, MEGDNN_MAX_NDIM, &ndim, padA, strideA, dilationA, &mode, &computeType);
            ser.write_plain(ndim);
            for (int i = 0; i < ndim; ++i) {
                ser.write_plain(padA[i]);
                ser.write_plain(strideA[i]);
                ser.write_plain(dilationA[i]);
            }
            ser.write_plain(mode);
            ser.write_plain(computeType);
            return Empty{};
        }
        Empty deserialize(StringSerializer& ser, Empty) {
            int ndim = ser.read_plain<int>();
            int padA[MEGDNN_MAX_NDIM];
            int strideA[MEGDNN_MAX_NDIM];
            int dilationA[MEGDNN_MAX_NDIM];
            for (int i = 0; i < ndim; ++i) {
                padA[i] = ser.read_plain<int>();
                strideA[i] = ser.read_plain<int>();
                dilationA[i] = ser.read_plain<int>();
            }
            cudnnConvolutionMode_t mode = ser.read_plain<cudnnConvolutionMode_t>();
            cudnnDataType_t computeType = ser.read_plain<cudnnDataType_t>();
            cudnnSetConvolutionNdDescriptor(value, ndim, padA, strideA, dilationA, mode, computeType);
            return Empty{};
        }
    };
    class CudnnTensorDescParam {
    public:
        cudnnTensorDescriptor_t value;
        Empty serialize(StringSerializer& ser, Empty) {
            int nbDims = MEGDNN_MAX_NDIM;
            cudnnDataType_t dataType;
            int dimA[MEGDNN_MAX_NDIM];
            int strideA[MEGDNN_MAX_NDIM];
            cudnnGetTensorNdDescriptor(value, nbDims, &dataType, &nbDims, dimA, strideA);
            ser.write_plain(nbDims);
            for (int i = 0; i < nbDims; ++i) {
                ser.write_plain(dimA[i]);
                ser.write_plain(strideA[i]);
            }
            ser.write_plain(dataType);
            return Empty{};
        }
        Empty deserialize(StringSerializer& ser, Empty) {
            int nbDims = MEGDNN_MAX_NDIM;
            cudnnDataType_t dataType;
            int dimA[MEGDNN_MAX_NDIM];
            int strideA[MEGDNN_MAX_NDIM];
            nbDims = ser.read_plain<int>();
            for (int i = 0; i < nbDims; ++i) {
                dimA[i] = ser.read_plain<int>();
                strideA[i] = ser.read_plain<int>();
            }
            dataType = ser.read_plain<cudnnDataType_t>();
            cudnnSetTensorNdDescriptor(value, dataType, nbDims, dimA, strideA);
            return Empty{};
        }
    };
    class CudnnFilterDescParam {
    public:
        cudnnFilterDescriptor_t value;
        Empty serialize(StringSerializer& ser, Empty) {
            int nbDims = MEGDNN_MAX_NDIM;
            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int filterDimA[MEGDNN_MAX_NDIM];
            cudnnGetFilterNdDescriptor(value, nbDims, &dataType, &format, &nbDims, filterDimA);
            ser.write_plain(nbDims);
            for (int i = 0; i < nbDims; ++i) {
                ser.write_plain(filterDimA[i]);
            }
            ser.write_plain(dataType);
            ser.write_plain(format);
            return Empty{};
        }
        Empty deserialize(StringSerializer& ser, Empty) {
            int nbDims = MEGDNN_MAX_NDIM;
            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int filterDimA[MEGDNN_MAX_NDIM];
            nbDims = ser.read_plain<int>();
            for (int i = 0; i < nbDims; ++i) {
                filterDimA[i] = ser.read_plain<int>();
            }
            dataType = ser.read_plain<cudnnDataType_t>();
            format = ser.read_plain<cudnnTensorFormat_t>();
            cudnnSetFilterNdDescriptor(value, dataType, format, nbDims, filterDimA);
            return Empty{};
        }
    };
}
