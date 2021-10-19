/**
 * \file src/opr/include/megbrain/opr/dnn/correlation.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

MGB_DEFINE_OPR_CLASS(
        CorrelationForward, intl::MegDNNOprWrapperFwd<megdnn::CorrelationForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC CorrelationForward(
            VarNode* data1, VarNode* data2, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar data1, SymbolVar data2, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using Correlation = CorrelationForward;

MGB_DEFINE_OPR_CLASS(
        CorrelationBackwardData1,
        intl::MegDNNOprWrapperBwd<megdnn::CorrelationBackwardData1>) // {
public:
    MGE_WIN_DECLSPEC_FUC CorrelationBackwardData1(
            VarNode* diff, VarNode* data1, VarNode* data2, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar diff, SymbolVar data1, SymbolVar data2, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void scn_do_execute() override;
    size_t get_workspace_size_bytes(
            const TensorShapeArray& input_shapes,
            const TensorShapeArray& output_shapes) const override;
};

MGB_DEFINE_OPR_CLASS(
        CorrelationBackwardData2,
        intl::MegDNNOprWrapperBwd<megdnn::CorrelationBackwardData2>) // {
public:
    MGE_WIN_DECLSPEC_FUC CorrelationBackwardData2(
            VarNode* diff, VarNode* data1, VarNode* data2, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar diff, SymbolVar data1, SymbolVar data2, const Param& param = {},
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
