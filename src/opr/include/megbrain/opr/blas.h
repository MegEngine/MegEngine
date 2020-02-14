/**
 * \file src/opr/include/megbrain/opr/blas.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/exception.h"
#include "megbrain/tensor.h"
#include "megbrain/graph.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include "megdnn/oprs/linalg.h"

namespace mgb {
namespace opr {

/*!
 * \brief matrix_mul(trans0(opr0), trans1(opr1))
 */
MGB_DEFINE_OPR_CLASS(MatrixMul,
        intl::MegDNNOprWrapperFwd<megdnn::MatrixMul>) // {

    public:

        MatrixMul(VarNode *opr0, VarNode *opr1,
                const Param &param, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar opr0, SymbolVar opr1,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
    private:
        void add_input_layout_constraint() override;
        void scn_do_execute() override;
        void init_output_dtype() override;
        size_t get_workspace_size_bytes(
                const TensorShapeArray &input_shapes,
                const TensorShapeArray &output_shapes) const override;

        static bool check_layout(const TensorLayout &layout, int transpose);
};

/*!
 * \brief batched matrix multiplication on 3D inputs
 */
MGB_DEFINE_OPR_CLASS(BatchedMatrixMul,
        intl::MegDNNOprWrapperFwd<megdnn::BatchedMatrixMul>) // {

    public:

        BatchedMatrixMul(VarNode *opr0, VarNode *opr1,
                const Param &param, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar opr0, SymbolVar opr1,
                const Param &param = {},
                const OperatorNodeConfig &config = {});
    private:
        void add_input_layout_constraint() override;
        void init_output_dtype() override;
        void scn_do_execute() override;
        size_t get_workspace_size_bytes(
                const TensorShapeArray &input_shapes,
                const TensorShapeArray &output_shapes) const override;

        static bool check_layout(const TensorLayout &layout, bool transpose);
};

/*!
 * \brief dot product of two tensors
 */
MGB_DEFINE_OPR_CLASS(Dot, cg::SingleCNOperatorNodeBaseT<
        mixin::MegDNNOprHolderImpl<megdnn::Dot>>) // {

    public:
        Dot(VarNode *opr0, VarNode *opr1, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar opr0, SymbolVar opr1,
                const OperatorNodeConfig &config = {});

        // for serialization
        static SymbolVar make(SymbolVar opr0, SymbolVar opr1, Param,
                const OperatorNodeConfig &config) {
            return make(opr0, opr1, config);
        }

    private:
        void add_input_layout_constraint() override;
        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        void record_execute_deps(ExecDependencyArray &deps) override;
};

MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD1(MatrixInverse);

MGB_DEFINE_OPR_CLASS(SVD, intl::MegDNNOprWrapperFwd<megdnn::SVD>) // {
    public:
        SVD(VarNode* src, const Param& param, const OperatorNodeConfig &config);
        static SymbolVarArray make(const SymbolVar& src, const Param &param = {},
                                   const OperatorNodeConfig& config = {});
};

} // opr
} // mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

