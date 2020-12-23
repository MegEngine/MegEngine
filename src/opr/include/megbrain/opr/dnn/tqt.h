/**
 * \file src/opr/include/megbrain/opr/dnn/tqt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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

MGB_DEFINE_OPR_CLASS(TQTForward,
                     intl::MegDNNOprWrapperFwd<megdnn::TQTForward>)  // {
public:
    TQTForward(VarNode* src, VarNode* scale, const Param& param,
           const OperatorNodeConfig& config);

    static SymbolVar make(SymbolVar src, SymbolVar scale, const Param& param = {},
                      const OperatorNodeConfig& config = {});
};
using TQT = TQTForward;

MGB_DEFINE_OPR_CLASS(TQTBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::TQTBackward>)  // {
public:
    TQTBackward(VarNode* y_grad, VarNode* x, VarNode* scale, const Param& param,
            const OperatorNodeConfig& config);

    static SymbolVarArray make(SymbolVar y_grad, SymbolVar x, SymbolVar scale,
                           const Param& param = {},
                           const OperatorNodeConfig& config = {});

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
};

}  // namespace opr
}  // namespace mgb
