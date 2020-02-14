/**
 * \file src/jit/include/megbrain/jit/placeholder_opr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"

#if MGB_JIT

namespace mgb {
namespace jit {

/*!
 * \brief placeholder opr for compilation cache
 *
 * JITPlaceholder oprs act as the leaves (i.e. input nodes) of the
 * InternalGraph. They are canonized to have no shape infer and cpu:default comp
 * node, so the same set of JITPlaceholder oprs can be shared among multiple JIT
 * expressions. Therefore identity of the JIT expression can be checked by
 * directly comparing the final var nodes due to graph deduplication.
 *
 * Note that the oprs in the sub graph are only used for representing the
 * computing graph before being lowered to an AST, and they should not get
 * involved in any actual computing.
 */
MGB_DEFINE_OPR_CLASS(JITPlaceholder, cg::SingleCNOperatorNodeBase) // {
public:
    //! input type of this JITPlaceholder
    enum class InpType {
        DEV_VALUE,             //!< tensor value on computing device
        HOST_VALUE_FOR_SHAPE,  //!< a tensor shape constructed from statically
                               //!< inferred value
    };

    JITPlaceholder(VarNode* src_var, size_t id, InpType inp_type);

    /*!
     * \param src_var the original var that provides dtype and owner graph
     * \param id id of this placeholder in the sub graph
     * \param inp_type input type of this placeholder as described in InpType
     */
    static SymbolVar make(VarNode* src_var, size_t id,
                          InpType inp_type = InpType::DEV_VALUE);

    //! index of this var in the inputs of the JIT opr
    size_t input_id() const { return m_id; }

    InpType inp_type() const { return m_inp_type; }

    bool is_host_value_shape_input() const {
        return inp_type() == InpType::HOST_VALUE_FOR_SHAPE;
    }

private:
    void init_output_static_infer_desc() override;
    void scn_do_execute() override;
    void init_output_comp_node() override;

    const InpType m_inp_type;
    const size_t m_id;
};

using PlaceholderArray = SmallVector<JITPlaceholder*>;

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
