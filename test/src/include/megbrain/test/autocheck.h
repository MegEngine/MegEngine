/**
 * \file test/src/include/megbrain/test/autocheck.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief automatically check operator and grad
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/test/helper.h"

#include <array>

namespace mgb {

/*!
 * \brief check single cn opr forward and numerical grad
 *
 * The input/output tensors would have the same dtype and comp node
 *
 * \tparam dtype default input data type; also the output data type
 */
template<size_t nr_inp, size_t nr_out, class dtype = dtype::Float32>
class AutoOprChecker {
    public:
        using ctype = typename DTypeTrait<dtype>::ctype;
        using SymInpArray = std::array<cg::SymbolVar, nr_inp>;
        using SymOutArray = std::array<cg::SymbolVar, nr_out>;
        using NumInpArray = std::array<std::shared_ptr<HostTensorND>, nr_inp>;
        using NumOutArray = std::array<HostTensorND, nr_out>;
        using ShapeInpArray = std::array<TensorShape, nr_inp>;

        using GraphMaker = std::function<SymOutArray(const SymInpArray&)>;
        using FwdNumeric = std::function<
            void(NumOutArray&, const NumInpArray &)>;

        //! callback to generate one input value by filling in dest
        using InputGenerator = std::function<void(HostTensorND &dest)>;

        //! callback to modify generated inputs together so they can have some
        //! property
        using InputCoordinator = std::function<void(const NumInpArray&)>;

        //! callback to get input dump message on error
        using InputDumpOnError = std::function<std::string(const NumInpArray&)>;

        //! callback to inspect computing sequences
        using CallbackCompSeq = std::function<
            void(ComputingGraph *, cg::AsyncExecutable*)>;

        AutoOprChecker(GraphMaker maker, FwdNumeric fwd,
                CompNode comp_node = CompNode::load("xpu0"));

        struct RunOptions {
            //! contiguous loss coefficient, used for debug
            bool cont_loss_p = false;
            float outputs_max_err = 1e-5,
                  numdiff_eps = 1e-2,
                  numdiff_max_err = 1e-3;

            //! override numdiff set for a single input
            std::array<Maybe<float>, nr_inp>
                numdiff_eps_single_inp, numdiff_max_err_single_inp;
        };

        AutoOprChecker& set_input_generator(size_t idx,
                const InputGenerator &gen);

        AutoOprChecker& set_input_coordinator(const InputCoordinator &coord);

        AutoOprChecker& set_input_allow_grad(size_t idx,
                bool allowed);

        AutoOprChecker& set_input_default_shape(size_t idx,
                const TensorShape &shape);

        AutoOprChecker& set_output_allow_grad(size_t idx,
                bool allowed);

        AutoOprChecker& set_output_allow_check(size_t idx,
                bool allowed);

        AutoOprChecker& set_input_dump_on_error(InputDumpOnError dump) {
            m_inp_dump_on_error = dump;
            return *this;
        }

        AutoOprChecker& disable_grad_check() {
            m_need_grad_check = false;
            return *this;
        }

        AutoOprChecker& disable_multi_loss_check() {
            m_need_multi_loss_check = false;
            return *this;
        }

        AutoOprChecker& disable_graph_opt() {
            mgb_assert(m_graph == nullptr, "cannot disable graph optimization "
                       "after graph is built");
            m_disable_graph_opt = true;
            return *this;
        }

        //! insert a callback after grads value have been computed
        AutoOprChecker& on_grad_computed(const CallbackCompSeq &cb) {
            m_on_grad_computed = cb;
            return *this;
        }

        AutoOprChecker& run(const ShapeInpArray &shapes,
                const RunOptions &opt = {});

        //! set extra message to be outputed on error
        AutoOprChecker& set_extra_err_msg(std::string msg) {
            m_extra_err_msg = std::move(msg);
            return *this;
        }

        //! change the dtype of a single input
        AutoOprChecker& set_input_dtype(size_t idx, DType dtype_) {
            *m_inputs.at(idx) = HostTensorND{m_comp_node, dtype_};
            return *this;
        }

        //! set whether to use virtual grad
        AutoOprChecker& set_use_virtual_grad(bool use_virtual_grad) {
            m_use_virtual_grad = use_virtual_grad;
            return *this;
        }

        //! disable loss / grad seperate compile in the third check
        AutoOprChecker& disable_check_loss_grad_seperate_compile() {
            m_disable_check_loss_grad_seperate_compile = true;
            return *this;
        }

        //! when enable virtual_grad, grad depends on the actual loss's stream,
        //! may lead to different streams when executing the second iteration
        AutoOprChecker& clean_grad_cn() {
            mgb_assert(m_use_virtual_grad);
            for (auto&& grad : m_grads) {
                grad = {};
            }
            for (auto&& grad : m_grads_mul2) {
                grad = {};
            }
            return *this;
        }

        //! get the comp node to run the checker
        CompNode comp_node() const { return m_comp_node; }

        ~AutoOprChecker();

    private:
        bool m_need_grad_check =
            DTypeTrait<dtype>::category == DTypeCategory::FLOAT;

        bool m_built = false;
        bool m_failed = false, m_should_copy_grad = true;
        bool m_need_multi_loss_check = true;
        bool m_disable_graph_opt = false;
        bool m_disable_check_loss_grad_seperate_compile = false;

        FwdNumeric m_fwd;
        GraphMaker m_maker;
        CompNode m_comp_node;
        bool m_use_virtual_grad = false;

        int m_run_cnt = 0;
        std::shared_ptr<cg::ComputingGraph> m_graph;
        std::unique_ptr<cg::AsyncExecutable> m_func;
        cg::ComputingGraph::OutputSpec m_outspec_fwd_grad, m_outspec_loss;
        InputDumpOnError m_inp_dump_on_error;

        // callbacks
        CallbackCompSeq m_on_grad_computed;

        // inputs
        HostTensorGenerator<dtype> m_gen;
        NumInpArray m_inputs;
        std::array<InputGenerator, nr_inp> m_inputs_generator;
        // specify if taking grads wrt inp is allowed
        std::array<bool, nr_inp> m_inputs_allow_grad;
        InputCoordinator m_input_coordinator;

        // outputs
        NumOutArray m_outputs, m_outputs_truth;
        // specify if taking grads wrt out is allowed
        std::array<bool, nr_out> m_outputs_allow_grad;
        // specify if we should compare the i-th output results
        std::array<bool, nr_out> m_outputs_allow_check;

        // loss and grads
        std::array<std::shared_ptr<HostTensorND>, nr_out> m_loss_p;
        std::array<HostTensorND, nr_inp> m_grads, m_grads_mul2;
        HostTensorND m_loss, m_cur_grad;

        // misc
        std::string m_extra_err_msg;

        void build_graph();

        void do_run(const ShapeInpArray &shapes, const RunOptions &opt);
};

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

