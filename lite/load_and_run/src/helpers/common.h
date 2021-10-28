/**
 * \file lite/load_and_run/src/helpers/common.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */

#pragma once
#include <gflags/gflags.h>
#include <memory>
DECLARE_int32(thread);
namespace lar {
/*!
 * \brief: state of model running
 */
enum class RunStage {

    BEFORE_MODEL_LOAD = 0,

    AFTER_MODEL_LOAD = 1,

    BEFORE_OUTSPEC_SET = 2,

    //! using for dump static memory information svg file
    AFTER_OUTSPEC_SET = 3,

    //! using for external c opr library
    MODEL_RUNNING = 4,

    //! using for output dumper
    AFTER_RUNNING_WAIT = 5,

    //! using for external c opr library
    AFTER_RUNNING_ITER = 6,

    AFTER_MODEL_RUNNING = 7,
};
/*!
 * \brief: type of different model
 */
enum class ModelType {
    LITE_MODEL = 0,
    MEGDL_MODEL,
    UNKNOWN,
};
/*!
 * \brief: param for running model
 */
struct RuntimeParam {
    RunStage stage = RunStage::AFTER_MODEL_LOAD;
    size_t warmup_iter;             //! warm up number before running model
    size_t run_iter;                //! iteration number for running model
    size_t threads = FLAGS_thread;  //! thread number for running model (NOTE:it's
                                    //! different from multithread device )
    size_t testcase_num = 1;        //! testcase number for model with testcase
};
/*!
 * \brief:layout type  for running model optimization
 */
enum class OptLayoutType {
    NCHW4 = 1 << 0,
    CHWN4 = 1 << 1,
    NCHW44 = 1 << 2,
    NCHW88 = 1 << 3,
    NCHW32 = 1 << 4,
    NCHW64 = 1 << 5,
    NHWCD4 = 1 << 6,
    NCHW44_DOT = 1 << 7
};

}  // namespace lar
// vim: syntax=cpp.doxygen
