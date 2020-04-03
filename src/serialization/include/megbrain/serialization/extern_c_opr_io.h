/**
 * \file src/serialization/include/megbrain/serialization/extern_c_opr_io.h
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
#include "megbrain/serialization/extern_c_opr.h"
#include "megbrain/serialization/opr_registry.h"

namespace mgb {
namespace opr {

//! an operator to run extern C oprs
MGB_DEFINE_OPR_CLASS(ExternCOprRunner,
                           cg::SingleCNOutshapePureByInshapeOprBase) // {
    std::shared_ptr<MGBOprDesc> m_desc;

    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    void scn_do_execute() override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;

    static cg::OperatorNodeBase* make_from_desc_shared(
            const VarNodeArray& inputs, std::shared_ptr<MGBOprDesc> desc,
            const OperatorNodeConfig& config);

public:
    ExternCOprRunner(const VarNodeArray& inputs,
                     std::shared_ptr<MGBOprDesc> desc,
                     const OperatorNodeConfig& config);

    //! create from MGBOprDesc and steal its reference
    static cg::OperatorNodeBase* make_from_desc(
            const VarNodeArray& inputs, MGBOprDesc* desc,
            const OperatorNodeConfig& config = {});

    /*!
     * \brief make a placeholder so this opr can be placed in the graph to
     *      produce a graph dump
     *
     * Note: this operator can not be executed
     *
     * \param output_shapes predefined output shapes
     * \param name operator dump name that should match the name in MGBOprLoader
     * \param data data to be written to file for dump
     * \param data_len length of \p data
     * \param output_dtypes predefined output dtypes
     */
    static cg::OperatorNodeBase* make_placeholder(
            const SymbolVarArray& inputs, const TensorShapeArray& output_shapes,
            const char* name, const void* data, size_t data_len,
            const OperatorNodeConfig& config = {},
            const SmallVector<DType>& output_dtypes = {});

    /*!
     * \brief unregister a MGBOprLoader
     * \return whether any loader is removed (i.e. whether the name exists)
     */
    static bool unregister_loader(const char* name);

    //! impl for serialization dump
    static void dump(serialization::OprDumpContext& ctx,
                     const cg::OperatorNodeBase& opr);

    //! impl for serialization load
    static cg::OperatorNodeBase* load(serialization::OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config);

    //! impl for serialization shallow copy
    static cg::OperatorNodeBase* shallow_copy(
            const serialization::OprShallowCopyContext& ctx,
            const cg::OperatorNodeBase& opr, const VarNodeArray& inputs,
            const OperatorNodeConfig& config);

    //! helper for converting TensorShape to MGBTensorShape
    static ::MGBTensorShape tensor_shape_to_c(const TensorShape& shape);

    //! helper for converting MGBTensorShape to TensorShape
    static TensorShape tensor_shape_from_c(const MGBTensorShape& shape);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
