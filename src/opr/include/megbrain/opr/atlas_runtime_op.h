/**
 * \file src/opr/include/megbrain/opr/atlas_runtime_op.h
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

#include <memory>
#include "megbrain/comp_node_env.h"
#include "megbrain/graph.h"
#include "megbrain/serialization/file.h"

#if MGB_ATLAS
#include "acl/acl.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(AtlasRuntimeOpr,
                     cg::SingleCNOutshapePureByInshapeOprBase) // {
public:
    using SharedBuffer = mgb::serialization::SharedBuffer;

    enum AippInputFormat {NO_AIPP, YUV420SP_U8, RGB888_U8};

    void scn_do_execute() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    void add_input_layout_constraint() override;
    void init_output_dtype() override;

    /**
     * \brief create AtlasRuntimeOpr with buf or set model with
     * a existance model.
     *
     * \brief Neither buf is set or model_id&model_desc is set
     */
    AtlasRuntimeOpr(SharedBuffer buf,
                    const std::pair<uint32_t, aclmdlDesc*>& model,
                    const VarNodeArray& inputs,
                    const OperatorNodeConfig& config);
    ~AtlasRuntimeOpr();

    const SharedBuffer& buffer() const { return m_buffer; }

    std::pair<uint32_t, aclmdlDesc*> model() const {
        return {m_model_id, m_model_desc};
    }

    static SymbolVarArray make(SharedBuffer buf, const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

    static SymbolVarArray make(SharedBuffer buf,
                               const std::pair<uint32_t, aclmdlDesc*>& model,
                               const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

    static SymbolVarArray make(const void* buf, size_t size,
                               const SymbolVarArray& src,
                               const OperatorNodeConfig& config = {});

private:
    SharedBuffer m_buffer;
    constexpr static uint32_t INVALID_MODEL_ID = -1;
    uint32_t m_model_id = INVALID_MODEL_ID;
    aclmdlDesc* m_model_desc = nullptr;
    //! if set true, it will release model
    bool m_is_model_holder = false;
    SmallVector<AippInputFormat> m_aipp_input_format;
    //! Atlas need a 64bit device tensor to hold dynamic batch state
    DeviceTensorND m_dyn_batch_tensor;
    SmallVector<size_t> m_dyn_batch_choices;
};

}  // namespace opr
}  // namespace mgb

#endif  // MGB_ATLAS

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
