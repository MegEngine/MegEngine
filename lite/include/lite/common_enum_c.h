/**
 * \file inlude/lite/common_enum_c.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef LITE_COMMON_ENUM_C_H_
#define LITE_COMMON_ENUM_C_H_

/*!
 * \brief The log level.
 */
typedef enum LiteLogLevel {
    DEBUG = 0, /*!< The lowest level and most verbose */
    INFO = 1,  /*!< The lowest level and most verbose */
    WARN = 2,  /*!< Print only warning and errors */
    ERROR = 3, /*!< Print only errors */
} LiteLogLevel;

typedef enum LiteBackend {
    LITE_DEFAULT = 0, //! default backend is mge
} LiteBackend;

typedef enum LiteDeviceType {
    LITE_CPU = 0,
    LITE_CUDA = 1,
    LITE_ATLAS = 3,
    LITE_NPU = 4,
    //! when the device information is set in model, so set LITE_DEVICE_DEFAULT
    //! in lite
    LITE_DEVICE_DEFAULT = 5,
} LiteDeviceType;

typedef enum LiteDataType {
    LITE_FLOAT = 0,
    LITE_HALF = 1,
    LITE_INT = 2,
    LITE_INT16 = 3,
    LITE_INT8 = 4,
    LITE_UINT8 = 5,
    LITE_UINT = 6,
    LITE_UINT16 = 7,
    LITE_INT64 = 8,
} LiteCDataType;

typedef enum LiteTensorPhase {
    //! Tensor maybe input or output
    LITE_IO = 0,
    //! Tensor is input
    LITE_INPUT = 1,
    //! Tensor is output
    LITE_OUTPUT = 2,
} LiteTensorPhase;

/*!
 * \brief the input and output type, include SHAPE and VALUE
 * sometimes user only need the shape of the output tensor
 */
typedef enum LiteIOType {
    LITE_IO_VALUE = 0,
    LITE_IO_SHAPE = 1,
} LiteIOType;

/*!
 * \brief operation algorithm seletion strategy type, some operations have
 * multi algorithms, different algorithm has different attribute, according to
 * the strategy, the best algorithm will be selected.
 *
 * Note: These strategies can be combined
 *
 * 1. LITE_ALGO_HEURISTIC | LITE_ALGO_PROFILE means: if profile cache not valid,
 * use heuristic instead
 *
 * 2. LITE_ALGO_HEURISTIC | LITE_ALGO_REPRODUCIBLE means: heuristic choice the
 * reproducible algo
 *
 * 3. LITE_ALGO_PROFILE | LITE_ALGO_REPRODUCIBLE means: profile the best
 * algorithm from the reproducible algorithms set
 *
 * 4. LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED means: profile the best
 * algorithm form the optimzed algorithms, thus profile will process fast
 *
 * 5. LITE_ALGO_PROFILE | LITE_ALGO_OPTIMIZED | LITE_ALGO_REPRODUCIBLE means:
 * profile the best algorithm form the optimzed and reproducible algorithms
 */
typedef enum LiteAlgoSelectStrategy {
    LITE_ALGO_HEURISTIC = 1 << 0,
    LITE_ALGO_PROFILE = 1 << 1,
    LITE_ALGO_REPRODUCIBLE = 1 << 2,
    LITE_ALGO_OPTIMIZED = 1 << 3,
} LiteAlgoSelectStrategy;

#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
