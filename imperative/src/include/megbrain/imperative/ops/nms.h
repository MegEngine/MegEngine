/**
 * \file imperative/src/include/megbrain/imperative/ops/nms.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
