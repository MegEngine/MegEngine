/**
 * \file src/opr/include/megbrain/opr/nn_int.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"

#include "megdnn/oprs/nn_int.h"

namespace mgb {
namespace opr {

namespace intl {
using ElemwiseMultiTypeBase = cg::SingleCNOperatorNode<
        cg::OutshapePureByInshapeOpr<>,
        mixin::MegDNNOprHolderImpl<megdnn::ElemwiseMultiType, false>>;
}

MGB_DEFINE_OPR_CLASS(ElemwiseMultiType, intl::ElemwiseMultiTypeBase) // {
public:
    using Mode = Param::Mode;

    ElemwiseMultiType(const VarNodeArrayView& inputs, Param param,
                      const OperatorNodeConfig& config);

    static SymbolVar make(const VarNodeArrayView& inputs, Param param,
                          const OperatorNodeConfig& config = {});

private:
    using ModeTrait = megdnn::ElemwiseMultiType::ModeTrait;

    void scn_do_execute() override;

    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;

    void init_output_dtype() override;

    void record_execute_deps(ExecDependencyArray& deps) override;
};

//! deprecated; TODO: remove in megbrain 8
class AffineInt final : public DynTypeObj {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Param = megdnn::param::Empty;
    static SymbolVar make(SymbolVar x, SymbolVar k, SymbolVar b,
                          const Param& param = {},
                          const OperatorNodeConfig& config = {}) {
        return ElemwiseMultiType::make(
                {x, k, b},
                {ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8}, config);
    }

    static Param param() {
        mgb_trap();
        return {};
    }
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
