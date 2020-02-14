/**
 * \file src/tensorrt/test/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/test/helper.h"

#if MGB_ENABLE_TENSOR_RT

namespace mgb {
namespace tensorrt {
/*!
 * \brief helper class for testing fusions on specific funcs
 *
 * The tensorrt opr would be created based on automatic opr replace pass
 */
class TrtReplaceChecker {
public:
    using ExpFunc = thin_function<SymbolVar(const SymbolVarArray&)>;

    TrtReplaceChecker(size_t nr_input, ExpFunc exp_func, CompNode cn)
            : m_nr_input{nr_input},
              m_comp_node{cn},
              m_graph{ComputingGraph::make()},
              m_exp_func{std::move(exp_func)},
              m_epsilon{1e-5} {}

    //! set input data type, which is float32 by default
    TrtReplaceChecker& set_dtype(size_t idx, DType dtype) {
        m_idx2dtype[idx] = dtype;
        return *this;
    }

    //! set input rng generator, which is default generator of float32
    TrtReplaceChecker& set_rng_gen(size_t idx,
                                   HostTensorGeneratorBase* rng_gen) {
        m_idx2rng_gen[idx] = rng_gen;
        return *this;
    }

    //! set input is a const var node
    TrtReplaceChecker& set_const_var(size_t idx) {
        m_mark_inp_const.insert(idx);
        return *this;
    }

    //! set epsilon
    TrtReplaceChecker& set_epsilon(float epsilon) {
        m_epsilon = epsilon;
        return *this;
    }


    /*!
     * \brief run and check correctness
     *
     * The graph would be built (and m_exp_func is invoked) on first call.
     */
    TrtReplaceChecker& run(const TensorShapeArray& input_shapes);

private:
    const size_t m_nr_input;
    const CompNode m_comp_node;
    HostTensorGenerator<> m_input_gen;
    SmallVector<std::shared_ptr<HostTensorND>> m_inputs_val;
    //! first item is output; following are input grads
    std::tuple<HostTensorND, HostTensorND> m_output_val;
    ThinHashMap<size_t, DType> m_idx2dtype;
    ThinHashMap<size_t, HostTensorGeneratorBase*> m_idx2rng_gen; 
    ThinHashSet<size_t> m_mark_inp_const;
    std::shared_ptr<ComputingGraph> m_graph;
    std::unique_ptr<cg::AsyncExecutable> m_func;

    ExpFunc m_exp_func;
    SymbolVar m_truth_y, m_trt_y;
    float m_epsilon;

    //! init m_graph and related fields; m_inputs_val must have been initialized
    void ensure_init_graph();
};

}  // namespace tensorrt
}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
