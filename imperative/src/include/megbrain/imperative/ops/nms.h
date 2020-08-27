/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb::imperative {

class NMSKeep : public OpDefImplBase<NMSKeep> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    float iou_thresh;     //!< IoU threshold for overlapping
    uint32_t max_output;  //!< max number of output boxes per batch
    NMSKeep() = default;
    NMSKeep(float iou_thresh_, uint32_t max_output_):
        iou_thresh(iou_thresh_), max_output(max_output_) {}
};

} // namespace mgb::imperative
