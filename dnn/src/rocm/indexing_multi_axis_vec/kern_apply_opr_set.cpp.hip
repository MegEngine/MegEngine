/**
 * \file src/rocm/indexing_multi_axis_vec/kern_apply_opr_set.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "hip_header.h"
#include "src/common/indexing_multi_axis_vec_kdef.h"
#define KERN_APPLY_OPR_OPR  ::megdnn::indexing_multi_axis_vec_kdef::OprSet
#include "./kern_apply_opr_impl.hipinl"

// vim: ft=cuda syntax=cpp.doxygen

