/**
 * \file src/opr/include/megbrain/opr/dnn/pooling.h
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
#include "megbrain/opr/search_policy/algo_chooser_helper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(PoolingForward,
                     intl::MegDNNOprWrapperFwd<megdnn::PoolingForward>,
                     public mixin::AlgoChooserHelper) //{
public:
    PoolingForward(VarNode * src, const Param& param,
                   const ExecutionPolicy& policy,
                   const OperatorNodeConfig& config);
    static SymbolVar make(SymbolVar src, const Param& param,
                          const OperatorNodeConfig& config = {},
                          const ExecutionPolicy& policy = {});

    void init_output_static_infer_desc() override;

    size_t get_workspace_size_bytes(const TensorShapeArray& input_shapes,
                                    const TensorShapeArray& output_shapes)
            const override;
};
using Pooling = PoolingForward;

MGB_DEFINE_OPR_CLASS(PoolingBackward,
                     intl::MegDNNOprWrapperBwd<megdnn::PoolingBackward>,
                     public mixin::AlgoChooserHelper) //{
public:
    PoolingBackward(VarNode * src, VarNode * dst, VarNode * diff,
                    const Param& param, const ExecutionPolicy& policy,
                    const OperatorNodeConfig& config);
    
    static SymbolVar make(SymbolVar src, SymbolVar dst, SymbolVar diff,
                          const Param& param,
                          const OperatorNodeConfig& config = {},
                          const ExecutionPolicy& policy = {});

    size_t get_workspace_size_bytes(const TensorShapeArray& input_shapes,
                                    const TensorShapeArray& output_shapes)
            const override final;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
