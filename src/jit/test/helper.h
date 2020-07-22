/**
 * \file src/jit/test/helper.h
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

namespace mgb {
namespace jit {
enum class Backend { NONE, HALIDE, NVRTC, MLIR };

void set_backend(Backend backend);

/*!
 * \brief reverse topological order, starting from \p nd
 *
 * \param endpoints_set the oprs that should be assumed to have been visited
 */
std::vector<cg::OperatorNodeBase*> get_rev_topo_order(
        SymbolVar nd, ThinHashSet<VarNode*> endpoints_set = {});

/*!
 * \brief helper class for testing fusions on specific funcs
 *
 * The JIT opr would be created based on automatic fusion pass, and gradient
 * would be checked by taking its gradient, without further fusion.
 */
class FusionChecker {
public:
    using ExpFunc = thin_function<SymbolVar(const SymbolVarArray&)>;

    FusionChecker(size_t nr_input, ExpFunc exp_func, CompNode cn)
            : m_nr_input{nr_input},
              m_comp_node{cn},
              m_graph{ComputingGraph::make()},
              m_exp_func{std::move(exp_func)} {}

    //! set input data type, which is float32 by default
    FusionChecker& set_dtype(size_t idx, DType dtype) {
        m_idx2dtype[idx] = dtype;
        return *this;
    }

    //! disable gradient checking for all inputs
    FusionChecker& disable_inp_grad();

    //! build the JIT graph directly, without running an optimizer
    FusionChecker& enable_direct_build() {
        m_direct_build = true;
        return *this;
    }

    //! disable opr type check, only JITExecutor, Host2DeviceCopy and
    //! GetVarShape are among the whitelist.
    FusionChecker& disable_opr_type_check() {
        m_check_opr_type = false;
        return *this;
    }

    //! set jit level, default is 2, see graph_opt.jit in graph options
    //! for more details
    FusionChecker& set_jit_level(uint8_t jit_level) {
        m_jit_level = jit_level;
        return *this;
    }

    /*!
     * \brief run and check correctness
     *
     * The graph would be built (and m_exp_func is invoked) on first call.
     */
    FusionChecker& run(const TensorShapeArray& input_shapes);

private:
    bool m_check_opr_type = true;
    bool m_direct_build = false;
    const size_t m_nr_input;
    uint8_t m_jit_level = 2;
    const CompNode m_comp_node;
    HostTensorGenerator<> m_input_gen;
    SmallVector<std::shared_ptr<HostTensorND>> m_inputs_val;
    //! first item is output; following are input grads
    SmallVector<std::tuple<size_t, HostTensorND, HostTensorND>> m_outputs_val;
    ThinHashSet<size_t> m_disable_inp_grad;
    ThinHashMap<size_t, DType> m_idx2dtype;
    std::shared_ptr<ComputingGraph> m_graph;
    std::unique_ptr<cg::AsyncExecutable> m_func;

    ExpFunc m_exp_func;
    SymbolVar m_truth_y, m_jit_y;

    //! init m_graph and related fields; m_inputs_val must have been initialized
    void ensure_init_graph();
};

}  // namespace jit
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
