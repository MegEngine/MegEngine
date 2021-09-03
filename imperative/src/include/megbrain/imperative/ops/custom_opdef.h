/**
 * \file imperative/src/include/megbrain/imperative/ops/custom_opdef.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/custom/custom.h"
#include "megbrain/custom/manager.h"
#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

class CustomOpDef: public OpDefImplBase<CustomOpDef> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
    const std::shared_ptr<const custom::CustomOp> m_op;
    custom::Param m_param;
public:
    CustomOpDef(const std::shared_ptr<const custom::CustomOp> &op);
    CustomOpDef(const std::shared_ptr<const custom::CustomOp> &op,
                const custom::Param&);

    void param(const custom::Param&);
    custom::Param &param(void);
    custom::Param param(void) const;
    size_t input_num(void) const;
    size_t output_num(void) const;
    std::string name(void) const;
    custom::RunTimeId runtime_id(void) const;
    const std::shared_ptr<const custom::CustomOp> &impl(void) const;

    void compute(const SmallVector<DeviceTensorND>&, SmallVector<DeviceTensorND>*) const;
    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs(
        const SmallVector<TensorPtr> &inputs) const;
    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs(
        const SmallVector<LogicalTensorDesc>&) const;
};

class CustomOpDefFactory {
    custom::CustomOpManager *ops;
    CustomOpDefFactory();
public:
    PREVENT_COPY_AND_ASSIGN(CustomOpDefFactory);

    static CustomOpDefFactory *inst(void);
    static bool is_custom_op(const OpDef &op);

    std::vector<std::string> op_list(void) const;

    std::shared_ptr<OpDef> create_opdef(const std::string&) const;
    std::shared_ptr<OpDef> create_opdef(const custom::RunTimeId&) const;
    std::shared_ptr<OpDef> create_opdef(const std::string&, const custom::Param&) const;
    std::shared_ptr<OpDef> create_opdef(const custom::RunTimeId&, const custom::Param&) const;
};

}   // imperative
}   // mgb
