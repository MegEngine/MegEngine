/**
 * \file test/src/autocheck.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/test/autocheck.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"

#include "megbrain/test/numerical_diff.h"

#include <cmath>

using namespace mgb;

#define DEF_IMPL(_ret)                                   \
    template <size_t nr_inp, size_t nr_out, class dtype> \
    _ret AutoOprChecker<nr_inp, nr_out, dtype>

#define DEF_IMPL_CHAIN()                                 \
    template <size_t nr_inp, size_t nr_out, class dtype> \
    AutoOprChecker<nr_inp, nr_out, dtype>&               \
            AutoOprChecker<nr_inp, nr_out, dtype>

DEF_IMPL()::AutoOprChecker(GraphMaker maker, FwdNumeric fwd, CompNode comp_node)
        : m_fwd(fwd), m_maker(maker), m_comp_node{comp_node} {
    InputGenerator default_input_gen = [this](HostTensorND& dest) {
        dest = *m_gen(dest.shape(), m_comp_node);
    };
    for (size_t i = 0; i < nr_inp; ++i) {
        m_inputs[i] = std::make_shared<HostTensorND>(m_comp_node, dtype());
        m_inputs_generator[i] = default_input_gen;
    }
    for (size_t i = 0; i < nr_inp; ++i) {
        m_inputs_allow_grad[i] = true;
    }
    for (size_t i = 0; i < nr_out; ++i) {
        m_outputs_allow_grad[i] = true;
    }
    for (size_t i = 0; i < nr_out; ++i) {
        m_outputs_allow_check[i] = true;
    }
}

DEF_IMPL(void)::build_graph() {
    mgb_assert(!m_built);
    m_built = true;
    m_graph = ComputingGraph::make();
    auto&& graph = m_graph;
    if (m_disable_graph_opt) {
        graph->options().graph_opt_level = 0;
    }

    SymInpArray sym_in;

    SymbolVar one, zero;
    {
        HostTensorND tmp{m_comp_node, mgb::dtype::Float32()};
        auto p = tmp.resize({1}).ptr<float>();
        p[0] = 1;
        one = opr::SharedDeviceTensor::make(*graph, tmp, {"one"});
        p[0] = 0;
        zero = opr::SharedDeviceTensor::make(*graph, tmp, {"zero"});
    }

    for (size_t i = 0; i < nr_inp; ++i) {
        // to trigger graph trans
        sym_in[i] = opr::Host2DeviceCopy::make(*graph, m_inputs[i],
                                               ssprintf("inp%zu", i));
        auto dt = sym_in[i].dtype();
        auto a = opr::TypeCvt::make(one, dt), b = opr::TypeCvt::make(zero, dt);
        sym_in[i] = sym_in[i] * a + b;
    }

    m_failed = true;
    auto sym_out = m_maker(sym_in);
    m_failed = false;

    for (size_t i = 0; i < nr_out; ++i) {
        m_outputs_truth[i].comp_node(m_comp_node).dtype(sym_out[i].dtype());
        m_outspec_fwd_grad.push_back(
                make_callback_copy(sym_out[i], m_outputs[i]));
    }

    if (!m_need_grad_check)
        return;

    SymbolVar loss;
    bool first_loss = true;
    for (size_t i = 0; i < nr_out; ++i) {
        if (m_outputs_allow_grad[i]) {
            m_loss_p[i] = std::make_shared<HostTensorND>(m_comp_node, dtype());
            auto cur = opr::Dot::make(
                    sym_out[i].flatten(),
                    opr::Host2DeviceCopy::make(*graph, m_loss_p[i],
                                               ssprintf("lossp%zu", i)));
            if (first_loss) {
                loss = cur;
            } else {
                loss = loss + cur;
            }
            first_loss = false;
        }
    }

    if (first_loss) {
        m_need_grad_check = false;
        return;
    }

    auto make_grad = [&](SymbolVar target, SymbolVar wrt) {
        if (m_use_virtual_grad)
            return opr::VirtualGrad::make(target, wrt);
        else
            return cg::grad(target, wrt);
    };

    auto loss2 = loss * 2;
    m_outspec_loss.push_back({make_callback_copy(loss, m_loss)});
    for (size_t i = 0; i < nr_inp; ++i)
        if (m_inputs_allow_grad[i]) {
            SymbolVar g = make_grad(loss, sym_in[i]);
            auto cb = [this, i](DeviceTensorND& dev) {
                if (m_should_copy_grad)
                    m_grads[i].copy_from(dev).sync();
            };
            m_outspec_fwd_grad.push_back({g, cb});

            // test grad with a different loss var
            if (m_need_multi_loss_check) {
                auto g2 = make_grad(loss2, sym_in[i]);
                auto cb2 = [this, i](DeviceTensorND& dev) {
                    if (m_should_copy_grad)
                        m_grads_mul2[i].copy_from(dev).sync();
                };
                m_outspec_fwd_grad.push_back({g2, cb2});
            }
        }
}

DEF_IMPL()::~AutoOprChecker() {
    mgb_assert(m_failed || m_run_cnt >= 3,
               "less than 3 runs for autocheker; some paths not taken");
}

DEF_IMPL_CHAIN()::set_input_generator(size_t idx, const InputGenerator& gen) {
    mgb_assert(!m_built, "cannot set_input_generator after the first run");
    mgb_assert(idx < nr_inp);
    m_inputs_generator[idx] = gen;
    return *this;
}

DEF_IMPL_CHAIN()::set_input_coordinator(const InputCoordinator& coord) {
    mgb_assert(!m_built, "cannot set_input_generator after the first run");
    m_input_coordinator = coord;
    return *this;
}

DEF_IMPL_CHAIN()::set_input_allow_grad(size_t idx, bool allowed) {
    mgb_assert(!m_built, "cannot set_input_allow_grad after the first run");
    mgb_assert(idx < nr_inp);
    m_inputs_allow_grad[idx] = allowed;
    return *this;
}

DEF_IMPL_CHAIN()::set_input_default_shape(size_t idx,
                                          const TensorShape& shape) {
    mgb_assert(!m_built, "cannot set_input_allow_grad after the first run");
    mgb_assert(idx < nr_inp);
    m_inputs[idx]->resize(shape);
    return *this;
}

DEF_IMPL_CHAIN()::set_output_allow_grad(size_t idx, bool allowed) {
    mgb_assert(!m_built, "cannot set_output_allow_grad after the first run");
    mgb_assert(idx < nr_out);
    m_outputs_allow_grad[idx] = allowed;
    return *this;
}

DEF_IMPL_CHAIN()::set_output_allow_check(size_t idx, bool allowed) {
    mgb_assert(!m_built, "cannot set_output_allow_check after the first run");
    mgb_assert(idx < nr_out);
    m_outputs_allow_check[idx] = allowed;
    return *this;
}

DEF_IMPL(void)::do_run(const ShapeInpArray& shapes, const RunOptions& opt) {
    mgb_assert(m_built);

    auto failstr = [&](const std::string& type) {
        std::string ishp_str;
        for (auto&& i : shapes) {
            if (!ishp_str.empty())
                ishp_str.append(", ");
            ishp_str.append(i.to_string());
        }
        std::string msg = ssprintf("%s failed: input shapes: [%s]",
                                   type.c_str(), ishp_str.c_str());
        if (m_inp_dump_on_error) {
            std::string extra_msg = m_inp_dump_on_error(m_inputs);
            if (!extra_msg.empty()) {
                msg.append("\nextra message:\n");
                msg.append(extra_msg);
            }
        }
        if (!m_extra_err_msg.empty()) {
            msg.append("\nextra message: ");
            msg.append(m_extra_err_msg);
        }
        return msg;
    };

    m_failed = true;

    // gen input data
    for (size_t i = 0; i < nr_inp; ++i) {
        m_inputs[i]->resize(shapes[i]);
        m_inputs_generator[i](*m_inputs[i]);
        mgb_assert(m_inputs[i]->shape().eq_shape(shapes[i]));
    }
    if (MGB_GETENV("MGB_AUTOCHECK_DUMP_INPUT")) {
        static size_t run_id;
        auto fname = output_file(ssprintf("autocheck-inp-%zu.bin", run_id++));
        for (size_t i = 0; i < nr_inp; ++i) {
            write_tensor_to_file(*m_inputs[i], fname.c_str(), i ? 'a' : 'w');
        }
        mgb_log("autocheck: %zu input tensors written to %s", nr_inp,
                fname.c_str());
    }
    if (m_input_coordinator)
        m_input_coordinator(m_inputs);

    // forward for groundtruth
    m_fwd(m_outputs_truth, m_inputs);
    for (auto&& i : m_outputs_truth) {
        i.comp_node().sync();
    }

    // gen loss_p
    if (m_need_grad_check) {
        float cur_loss_v = 0;
        for (size_t i = 0; i < nr_out; ++i) {
            if (m_outputs_allow_grad[i]) {
                auto nr = m_outputs_truth[i].shape().total_nr_elems();
                mgb_assert(nr, "got empty output");
                if (opt.cont_loss_p) {
                    m_loss_p[i]->resize({nr});
                    auto ptr = m_loss_p[i]->template ptr<float>();
                    for (size_t j = 0; j < nr; ++j)
                        ptr[j] = ++cur_loss_v;
                } else {
                    *m_loss_p[i] = *m_gen({nr}, m_comp_node);
                    auto ptr = m_loss_p[i]->template ptr<float>();
                    for (size_t j = 0; j < nr; ++j) {
                        auto v = ptr[j];
                        bool vsign = v > 0;
                        v = std::abs(v) + 0.1;
                        ptr[j] = vsign ? v : -v;
                    }
                }
            }
        }
    }

    /*
     * for each 3 consecutive runs:
     * 0 and 1: m_func generates loss and grads
     * 2: m_func generates only grads in fwd, and loss in numdiff
     *
     * This scheme is used for recompiling the function a few times, so more
     * problems can be exposed.
     */

    if (m_run_cnt % 3 == 0) {
        auto spec = m_outspec_loss;
        spec.insert(spec.end(), m_outspec_fwd_grad.begin(),
                    m_outspec_fwd_grad.end());
        m_func = m_graph->compile(spec);
    } else if (!m_disable_check_loss_grad_seperate_compile &&
               m_run_cnt % 3 == 2)
        m_func = m_graph->compile(m_outspec_fwd_grad);

    m_should_copy_grad = true;
    m_func->execute();
    m_should_copy_grad = false;
    if (m_on_grad_computed)
        m_on_grad_computed(m_graph.get(), m_func.get());

    for (size_t i = 0; i < nr_out; ++i) {
        if (m_outputs_allow_check[i]) {
            MGB_ASSERT_TENSOR_NEAR(m_outputs_truth[i], m_outputs[i],
                                   opt.outputs_max_err)
                    << failstr(ssprintf("output[%zu]", i));
        }
    }

    if (!m_need_grad_check) {
        m_failed = false;
        return;
    }

    std::vector<HostTensorND*> numgrad_inp(nr_inp);
    if (!m_disable_check_loss_grad_seperate_compile && m_run_cnt % 3 == 2)
        m_func = m_graph->compile(m_outspec_loss);
    for (size_t i = 0; i < nr_inp; ++i)
        if (m_inputs_allow_grad[i])
            numgrad_inp[i] = m_inputs[i].get();
        else
            numgrad_inp[i] = nullptr;

    auto cost_f = [this] {
        m_func->execute();
        mgb_assert(m_loss.shape().is_scalar());
        return m_loss.ptr<float>()[0];
    };

    std::vector<Maybe<float>> numdiff_eps;
    for (size_t i = 0; i < nr_inp; ++i) {
        if (m_inputs_allow_grad[i]) {
            float v = opt.numdiff_eps;
            auto&& sv = opt.numdiff_eps_single_inp[i];
            if (sv.valid())
                v = sv.val();
            numdiff_eps.push_back(v);
        } else {
            numdiff_eps.push_back(None);
        }
    }
    auto numgrad = numerical_diff_pt2(numgrad_inp, cost_f, numdiff_eps);

    auto mul2_inplace = [](HostTensorND& t) -> HostTensorND& {
        auto ptr = t.ptr<typename DTypeTrait<dtype>::ctype>();
        for (size_t j = 0, jt = t.layout().total_nr_elems(); j < jt; ++j) {
            ptr[j] *= 2;
        }
        return t;
    };

    for (size_t i = 0; i < nr_inp; ++i) {
        if (m_inputs_allow_grad[i]) {
            auto err = opt.numdiff_max_err;
            {
                auto&& se = opt.numdiff_max_err_single_inp[i];
                if (se.valid())
                    err = se.val();
            }
            MGB_ASSERT_TENSOR_NEAR(numgrad.at(i), m_grads[i], err)
                    << failstr(ssprintf("grad[%zu]", i));

            // check that grad2 == 2 * grad
            if (m_need_multi_loss_check) {
                MGB_ASSERT_TENSOR_NEAR(mul2_inplace(m_grads[i]),
                                       m_grads_mul2[i], err)
                        << failstr(ssprintf(
                                   "2 * grad[%zu] (grad with another loss var)",
                                   i));
            }
        }
    }
    m_failed = false;
}

DEF_IMPL_CHAIN()::run(const ShapeInpArray& shapes, const RunOptions& opt) {
    if (!m_built)
        build_graph();

    if (m_failed) {
        mgb_log_error("testcase not executed due to previous error");
        return *this;
    }

    do_run(shapes, opt);
    ++m_run_cnt;
    return *this;
}

namespace mgb {
// explicit instantialization
#define I(a, b)                                          \
    template class AutoOprChecker<a, b, dtype::Float32>; \
    template class AutoOprChecker<a, b, dtype::Int32>;

I(1, 1);
I(1, 2);
I(1, 3);
I(1, 4);
I(2, 1);
I(2, 2);
I(2, 4);
I(3, 1);
I(3, 2);
I(3, 3);
I(4, 1);
I(5, 1);
I(6, 1);

#undef I
}

TEST(TestAutoCheck, APlusB) {
    using Checker = AutoOprChecker<2, 1>;
    auto make_graph =
            [](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {inputs[0] + inputs[1] * inputs[1]};
    };
    auto fwd = [](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        DeviceTensorND i0, i1, tmp, out;
        i0.copy_from(*inp[0]);
        i1.copy_from(*inp[1]);
        auto opr = opr::intl::create_megdnn_opr<megdnn::Elemwise>(
                dest[0].comp_node());
        using Mode = opr::Elemwise::Mode;
        opr::Elemwise::perform(Mode::MUL, tmp, {i1, i1}, opr);
        opr::Elemwise::perform(Mode::ADD, out, {tmp, i0}, opr);
        dest[0].copy_from(out).sync();
    };
    Checker(make_graph, fwd)
            .run({TensorShape{2, 3}, TensorShape{2, 3}})
            .run({TensorShape{5, 2, 3}, TensorShape{5, 1, 1}})
            .run({TensorShape{2, 3, 4, 5}, TensorShape{1}});
}

#undef DEF_IMPL
#undef DEF_IMPL_CHAIN

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
