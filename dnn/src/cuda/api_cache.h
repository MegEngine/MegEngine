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
        constexpr int maxNbDims = CUDNN_DIM_MAX - 2;
        int nbDims = maxNbDims;
        int padA[maxNbDims];
        int strideA[maxNbDims];
        int dilationA[maxNbDims];
        cudnnConvolutionMode_t mode;
        cudnnDataType_t computeType;
        cudnnGetConvolutionNdDescriptor(value, maxNbDims, &nbDims, padA,
                                        strideA, dilationA, &mode,
                                        &computeType);
        ser.write_plain(nbDims);
        for (int i = 0; i < nbDims; ++i) {
            ser.write_plain(padA[i]);
            ser.write_plain(strideA[i]);
            ser.write_plain(dilationA[i]);
        }
        ser.write_plain(mode);
        ser.write_plain(computeType);
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
        cudnnGetTensorNdDescriptor(value, MEGDNN_MAX_NDIM, &dataType, &nbDims,
                                   dimA, strideA);
        ser.write_plain(nbDims);
        for (int i = 0; i < nbDims; ++i) {
            ser.write_plain(dimA[i]);
            ser.write_plain(strideA[i]);
        }
        ser.write_plain(dataType);
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
        cudnnGetFilterNdDescriptor(value, nbDims, &dataType, &format, &nbDims,
                                   filterDimA);
        ser.write_plain(nbDims);
        for (int i = 0; i < nbDims; ++i) {
            ser.write_plain(filterDimA[i]);
        }
        ser.write_plain(dataType);
        ser.write_plain(format);
        return Empty{};
    }
};

template <typename T>
class CudnnConvAlgoPerfParam {
public:
    T value;
    Empty serialize(StringSerializer& ser, Empty) {
        ser.write_plain(value.algo);
        ser.write_plain(value.status);
        ser.write_plain(value.time);
        ser.write_plain(value.memory);
        ser.write_plain(value.determinism);
        ser.write_plain(value.mathType);
        return Empty{};
    }

    Empty deserialize(StringSerializer& ser, Empty) {
        ser.read_plain(&value.algo);
        ser.read_plain(&value.status);
        ser.read_plain(&value.time);
        ser.read_plain(&value.memory);
        ser.read_plain(&value.determinism);
        ser.read_plain(&value.mathType);
        return Empty{};
    }
};
}  // namespace megdnn
