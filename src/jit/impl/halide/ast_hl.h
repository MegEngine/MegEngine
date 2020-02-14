/**
 * \file src/jit/impl/halide/ast_hl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./halide_header.h"

#if MGB_JIT_HALIDE

#include "megbrain/graph.h"
#include "megbrain/utils/metahelper.h"

namespace mgb {
namespace jit {
namespace ast_hl {

struct AstNode;
using AstNodePtr = std::shared_ptr<AstNode>;
using AstNodeArray = mgb::SmallVector<AstNodePtr>;

struct AstNode : public DynTypeObj {
    AstNodeArray m_inputs;

    //! halide func corresponding to the output var
    Halide::Func m_func;

    //! output layout
    megdnn::TensorLayout m_layout;

    virtual ~AstNode() = default;

    //! initialize m_func and m_layout according to func of the inputs
    virtual void init(cg::OperatorNodeBase* opr) = 0;
};

#define AST_NODE_DECL(_cls, _mem...)                   \
    struct _cls final : public AstNode {               \
        MGB_DYN_TYPE_OBJ_FINAL_DECL;                   \
                                                       \
    public:                                            \
        void init(cg::OperatorNodeBase* opr) override; \
        _mem;                                          \
    }

//! this is a special opr marking HOST_VALUE_FOR_SHAPE placeholders; its
//! m_inputs, m_func and m_layout are all empty
AST_NODE_DECL(InputHostValueShapeOp);

AST_NODE_DECL(InputDevValueOp, Halide::Buffer<> m_buffer);
AST_NODE_DECL(ElemwiseOp);
AST_NODE_DECL(TypeCvtOp);
AST_NODE_DECL(ReduceOp, Halide::Func m_comp);
AST_NODE_DECL(ScalarImmOp,
              union Val {
                  int32_t iv;
                  float fv;
              };
              Val m_val);
AST_NODE_DECL(BroadcastOp);

template <class Op>
inline Op* try_cast_as_op(AstNode* node) {
    if (node->same_type<Op>())
        return &node->cast_final<Op>();
    return nullptr;
}

//! make AstNodePtr from opr; no initialization is done
AstNodePtr make_from_opr(cg::OperatorNodeBase* opr);

}  // namespace ast_hl
}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_HALIDE

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
