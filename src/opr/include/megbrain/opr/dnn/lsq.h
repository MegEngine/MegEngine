/**
 * \file src/opr/include/megbrain/opr/dnn/lsq.h
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
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"
namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(LSQForward, intl::MegDNNOprWrapperFwd<megdnn::LSQForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSQForward(
            VarNode* src, VarNode* scale, VarNode* zero_point, VarNode* grad_scale,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar scale, SymbolVar zero_point, SymbolVar grad_scale,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSQ = LSQForward;

MGB_DEFINE_OPR_CLASS(
        LSQBackward, intl::MegDNNOprWrapperBwd<megdnn::LSQBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSQBackward(
            VarNode* y_grad, VarNode* x, VarNode* scale, VarNode* zero_point,
            VarNode* grad_scale, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar y_grad, SymbolVar x, SymbolVar scale, SymbolVar zero_point,
            SymbolVar grad_scale, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
};

}  // namespace opr
}  // namespace mgb
