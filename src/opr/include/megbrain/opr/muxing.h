/**
 * \file src/opr/include/megbrain/opr/muxing.h
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

namespace mgb {
namespace opr {

    /*!
     * \brief concat and then copy to all
     */
    MGB_DEFINE_OPR_CLASS(AllGather, cg::OutshapePureByInshapeOpr<>) // {

        class CopyStrategy;
        std::unique_ptr<CopyStrategy> m_copy_strategy;
        //! input layout corresponding to current copy strategy
        std::vector<TensorLayout> m_input_layout;

        int m_axis;

        void get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const override;
        void init_output_comp_node() override;
        void do_execute(ExecEnv &env) override;
        NodeProp* do_make_node_prop() const override;
        void on_mem_status_changed();
        OprEventCallback get_opr_event_callback() override final;

        void on_output_comp_node_stream_changed() override;

        public:
            AllGather(const VarNodeArray &input, int axis,
                    const OperatorNodeConfig &config);
            ~AllGather();

            VarNodeArray grad(const VarNodeArray &out_grad);

            static SymbolVarArray make(
                    const SymbolVarArray &input, int axis,
                    const OperatorNodeConfig &config = {});
    };

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

