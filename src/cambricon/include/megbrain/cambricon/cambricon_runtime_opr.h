/**
 * \file src/cambricon/include/megbrain/cambricon/cambricon_runtime_opr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node_env.h"
#include "megbrain/graph.h"
#include "megbrain/serialization/file.h"

#if MGB_CAMBRICON

namespace mgb {
namespace opr {
namespace cnrt_intl {
struct ModelUnloader {
    void operator()(cnrtModel_t* model) {
        if (model != nullptr)
            MGB_CNRT_CHECK(cnrtUnloadModel(*model));
    }
};
struct FunctionDeleter {
    void operator()(cnrtFunction_t* function) {
        if (function != nullptr)
            MGB_CNRT_CHECK(cnrtDestroyFunction(*function));
    }
};
struct RuntimeContextDeleter {
    void operator()(cnrtRuntimeContext_t* context) {
        if (context != nullptr)
            MGB_CNRT_CHECK(cnrtDestroyRuntimeContext(*context));
    }
};
using CnrtModelUniquePtr = std::unique_ptr<cnrtModel_t, ModelUnloader>;
using CnrtFunctionUniquePtr = std::unique_ptr<cnrtFunction_t, FunctionDeleter>;
using CnrtRuntimeContextUniquePtr =
        std::unique_ptr<cnrtRuntimeContext_t, RuntimeContextDeleter>;
};  // namespace cnrt_intl

MGB_DEFINE_OPR_CLASS(CambriconRuntimeOpr, cg::SingleCNOutshapePureByInshapeOprBase) // {
public:
    using CnrtModelUniquePtr = cnrt_intl::CnrtModelUniquePtr;
    using CnrtFunctionUniquePtr = cnrt_intl::CnrtFunctionUniquePtr;
    using CnrtRuntimeContextUniquePtr = cnrt_intl::CnrtRuntimeContextUniquePtr;
    using SharedBuffer = mgb::serialization::SharedBuffer;
    void scn_do_execute() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;

    CambriconRuntimeOpr(SharedBuffer buf, std::string symbol,
                        const VarNodeArray& inputs, bool tensor_dim_mutable,
                        const OperatorNodeConfig& config);

    const SharedBuffer& buffer() const {
        return m_buffer;
    }

    const std::string& symbol() const {
        return m_symbol;
    }

    bool is_tensor_dim_mutable() const {
        return m_tensor_dim_mutable;
    }

    static SymbolVarArray make(SharedBuffer buf, std::string symbol,
                               const SymbolVarArray& src,
                               bool tensor_dim_mutable = false,
                               const OperatorNodeConfig& config = {});

    static SymbolVarArray make(const void* buf, size_t size, std::string symbol,
                               const SymbolVarArray& src,
                               bool tensor_dim_mutable = false,
                               const OperatorNodeConfig& config = {});

private:
    SharedBuffer m_buffer; 
    std::string m_symbol;
    CnrtModelUniquePtr m_model;
    CnrtFunctionUniquePtr m_function;
    CnrtRuntimeContextUniquePtr m_context;
    bool m_tensor_dim_mutable;
};

}  // namespace opr
}  // namespace mgb

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

