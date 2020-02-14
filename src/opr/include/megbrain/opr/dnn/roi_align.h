/**
 * \file src/opr/include/megbrain/opr/dnn/roi_align.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(ROIAlignForward,
                           intl::MegDNNOprWrapperFwd<megdnn::ROIAlignForward>) // {
public:
    ROIAlignForward(VarNode* src, VarNode* rois, const Param& param,
                    const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar rois,
                          const Param& param = {},
                          const OperatorNodeConfig& config = {});
};
using ROIAlign = ROIAlignForward;

MGB_DEFINE_OPR_CLASS(
        ROIAlignBackward, intl::MegDNNOprWrapperBwd<megdnn::ROIAlignBackward>) // {
public:
    ROIAlignBackward(VarNode* diff, VarNode* src, VarNode* rois, VarNode* index,
                     const Param& param, const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar diff, SymbolVar src, SymbolVar rois,
                          SymbolVar index, const Param& param = {},
                          const OperatorNodeConfig& config = {});

private:
    void scn_do_execute() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
