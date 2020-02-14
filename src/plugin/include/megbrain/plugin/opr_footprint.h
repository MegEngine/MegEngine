/**
 * \file src/plugin/include/megbrain/plugin/opr_footprint.h
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

/*!
 * \brief footprint for operators
 */
class OprFootprint {
    //! function to calculate compution of a given operator
    using CompFootprintTrait = thin_function<uint64_t(cg::OperatorNodeBase*)>;
    ThinHashMap<Typeinfo*, CompFootprintTrait> m_type2comp_footprint;
#if MGB_ENABLE_JSON
    using ParamJsonTrait =
            thin_function<std::shared_ptr<json::Value>(cg::OperatorNodeBase*)>;
    ThinHashMap<Typeinfo*, ParamJsonTrait> m_type2param_json;
#endif

    //! add single footprint calculator for associated opr type.
    template <class OprType>
    void add_single_comp_footprint();

    //! add single param2json func for associated opr type
    template <class OprType>
    void add_single_param_json();

    //! be invoked when OprFootprint initilizing.
    void init_all_footprints();

public:
    struct Result {
        //! total input/output memory
        size_t memory = 0;

        //! total number of arithmetic computations; zero value means no trait
        //! function available
        uint64_t computation = 0;

        TensorLayoutArray inp_layout;
        TensorShapeArray out_shape;

        mgb::Typeinfo* opr_type;
#if MGB_ENABLE_JSON
        /*!
         * \brief param in json format
         */
        std::shared_ptr<json::Value> param;
        /*!
         * \brief convert this result to json object
         *
         * keys:
         *
         * computation
         * memory
         * in_shapes
         * out_shapes
         * in_layouts // only available if there are non-contig inputs
         */
        std::shared_ptr<json::Value> to_json() const;
#endif
    };

    OprFootprint() { init_all_footprints(); }

    //! return footprint rst for associated opr.
    Result calc_footprint(cg::OperatorNodeBase* opr);
    //! get computation of a given operator
    uint64_t get_computation(cg::OperatorNodeBase* opr);
#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> get_param_json(cg::OperatorNodeBase* opr);
    //! get opr foot print and graph exec info
    //! the function will recompile graph, AsyncExecutable compiled before will
    //! be invalid
    static std::shared_ptr<json::Value> get_opr_fp_graph_exec(
            cg::ComputingGraph& graph, const SymbolVarArray& outputs);
#endif
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
