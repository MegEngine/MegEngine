/**
 * \file src/cambricon/include/megbrain/cambricon/magicmind_runtime_opr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/serialization/file.h"

#if MGB_CAMBRICON

#include <sstream>
#include "interface_runtime.h"

#define MM_CHECK(stmt)                                                               \
    do {                                                                             \
        auto ret = (stmt);                                                           \
        if (ret != magicmind::Status::OK()) {                                        \
            std::ostringstream msg;                                                  \
            msg << ret;                                                              \
            mgb_throw(MegBrainError, "mm failure(extra msg:%s)", msg.str().c_str()); \
        }                                                                            \
    } while (0)

namespace mgb {
namespace opr {
namespace magicmind_intl {
template <typename T>
struct MagicMindDeleter {
    void operator()(T* p) {
        if (p != nullptr)
            p->Destroy();
    }
};

template <typename T>
using MagicMindUniquePtr = std::unique_ptr<T, MagicMindDeleter<T>>;
}  // namespace magicmind_intl

MGB_DEFINE_OPR_CLASS(
        MagicMindRuntimeOpr, cg::SingleCNOutshapePureByInshapeOprBase) // {
    void scn_do_execute() override;
    void get_output_var_shape(
            const TensorShapeArray& inp_shape,
            TensorShapeArray& out_shape) const override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;

public:
    template <typename T>
    using MagicMindUniquePtr = magicmind_intl::MagicMindUniquePtr<T>;
    //! Due to the requirement of shallow copy, the IModel should be shared among
    //! instances of magicmind operators.
    using IModelPtr = std::shared_ptr<magicmind::IModel>;
    using IContextPtr = MagicMindUniquePtr<magicmind::IContext>;
    using IEnginePtr = MagicMindUniquePtr<magicmind::IEngine>;
    class CambriconAllocator;
    using CambriconAllocatorPtr = std::shared_ptr<CambriconAllocator>;

    MagicMindRuntimeOpr(
            IModelPtr model, CambriconAllocatorPtr allocator,
            const VarNodeArray& inputs, const OperatorNodeConfig& config);

    //! get underlying inference model
    const IModelPtr& inference_model() const { return m_model; }

    //! get underlying cambricon allocator
    const CambriconAllocatorPtr& cambricon_allocator() const { return m_allocator; }

    //!
    static SymbolVarArray make(
            IModelPtr model, CambriconAllocatorPtr allocator, const SymbolVarArray& src,
            const OperatorNodeConfig& config);

    //! creator a magicmind runtime operator from a serialized memory buffer
    static SymbolVarArray make(
            const void* buf, size_t buf_size, const SymbolVarArray& src,
            const OperatorNodeConfig& config = {});

    static IModelPtr make_model_ptr(magicmind::IModel* model) {
        return {model, magicmind_intl::MagicMindDeleter<magicmind::IModel>()};
    }

private:
    CambriconAllocatorPtr m_allocator;
    IEnginePtr m_engine;
    mutable IContextPtr m_context;
    IModelPtr m_model;
};

}  // namespace opr
}  // namespace mgb

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
