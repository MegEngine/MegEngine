/**
 * \file src/opr/include/megbrain/opr/dnn/fake_quant.h
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
MGB_DEFINE_OPR_CLASS(FakeQuantForward,
                     intl::MegDNNOprWrapperFwd<megdnn::FakeQuantForward>) // {
public:
FakeQuantForward(VarNode* src, VarNode* scale, VarNode* zero_point,
                 const Param& param, const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar src, SymbolVar scale, SymbolVar zero_point,
                      const Param& param = {},
                      const OperatorNodeConfig& config = {});
};  // namespace opr
using FakeQuant = FakeQuantForward;

MGB_DEFINE_OPR_CLASS(FakeQuantBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::FakeQuantBackward>) // {
public:
FakeQuantBackward(VarNode* diff, VarNode* input, VarNode* scale,
                  VarNode* zero_point, const Param& param,
                  const OperatorNodeConfig& config);

static SymbolVar make(SymbolVar diff, SymbolVar input, SymbolVar scale,
                      SymbolVar zero_point, const Param& param = {},
                      const OperatorNodeConfig& config = {});

};

}  // namespace mgb
}  // namespace opr