/**
 * \file src/opr/include/megbrain/opr/tensor_gen.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"

#include "megdnn/oprs/general.h"

namespace mgb {
namespace opr {

/*!
 * \brief allocate a contiguous tensor of specified shape with undefined content
 */
MGB_DEFINE_OPR_CLASS(Alloc, intl::OutshapeBySymvarSCNOprBase) // {

    void outshape_by_symvar_do_get_output_shape(
            TensorShape &dest, const ShapeInferInfo &shpinfo) override;

    void scn_do_execute() override;
    public:
        Alloc(VarNode* shape, DType dtype, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar shape, DType dtype,
                const OperatorNodeConfig &config = {});

        static SymbolVar make(ComputingGraph &graph,
                const TensorShape &shape, DType dtype,
                const OperatorNodeConfig &config) {
            return make(
                    cg::var_from_tensor_shape(graph, config, "Alloc", shape),
                    dtype, config);
        }

        // for serialization
        using Param = megdnn::param::DType;
        static SymbolVar make(SymbolVar shape, Param param,
                const OperatorNodeConfig &config) {
            return make(shape, DType::from_enum(param.dtype), config);
        }
        Param param() const {
            return output(0)->dtype().enumv();
        }
};

MGB_DEFINE_OPR_CLASS(Linspace, cg::SingleCNOperatorNodeBase) // {

    public:
        using Param = megdnn::param::Linspace;

        Linspace(VarNode* start, VarNode *stop, VarNode *num,
                const Param &param, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar start, SymbolVar stop, SymbolVar num,
                const Param &param, const OperatorNodeConfig &config = {});

        const Param& param() const {
            return m_param;
        }

    private:
        const Param m_param;
        intl::UniqPtrWithCN<megdnn::Linspace> m_megdnn_opr;

        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        NodeProp* do_make_node_prop() const override;

        void record_execute_deps(ExecDependencyArray& deps) override;
};

MGB_DEFINE_OPR_CLASS(Eye, cg::SingleCNOperatorNodeBase) // {

    public:
        using Param = megdnn::Eye::Param;
        Eye(VarNode *shape,
                const Param &param, const OperatorNodeConfig &config);

        static SymbolVar make(SymbolVar shape,
                const Param &param, const OperatorNodeConfig &config = {});

        const Param& param() const {
            return m_param;
        }

    private:
        const Param m_param;
        intl::UniqPtrWithCN<megdnn::Eye> m_megdnn_opr;

        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        NodeProp* do_make_node_prop() const override;

        void record_execute_deps(ExecDependencyArray& deps) override;
};

} // opr
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

