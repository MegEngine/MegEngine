/**
 * \file imperative/src/test/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "helper.h"
#include "megbrain/graph.h"
#include "megbrain/opr/io.h"

#include <memory>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mgb {
namespace imperative {

namespace {

#define XSTR(s) STR(s)
#define STR(s) #s
#define CONCAT(a, b) a##b
#define PYINIT(name) CONCAT(PyInit_, name)
#define pyinit PYINIT(MODULE_NAME)

#define UNUSED __attribute__((unused))

extern "C" PyObject* pyinit();

class PyEnv {
    static std::unique_ptr<PyEnv> m_instance;
    std::unique_ptr<py::scoped_interpreter> m_interpreter;
    PyEnv();
public:
    static PyEnv& instance();
    static py::module get();
};

std::unique_ptr<PyEnv> PyEnv::m_instance = nullptr;

PyEnv::PyEnv() {
    mgb_assert(!m_instance);
    auto err = PyImport_AppendInittab(XSTR(MODULE_NAME), &pyinit);
    mgb_assert(!err);
    m_interpreter.reset(new py::scoped_interpreter());
}

PyEnv& PyEnv::instance() {
    if (!m_instance) {
        m_instance.reset(new PyEnv());
    }
    return *m_instance;
}

py::module PyEnv::get() {
    instance();
    return py::module::import(XSTR(MODULE_NAME));
}

py::array array(const Tensor& x) {
     PyEnv::get();
     return py::cast(x).attr("numpy")();
}

py::array array(const HostTensorND& x) {
    return array(*Tensor::make(x));
}

py::array array(const DeviceTensorND& x) {
    return array(*Tensor::make(x));
}

UNUSED void print(const Tensor& x) {
    return print(array(x));
}

UNUSED void print(const HostTensorND& x) {
    return print(array(x));
}

UNUSED void print(const DeviceTensorND& x) {
    return print(array(x));
}

UNUSED void print(const char* s) {
    PyEnv::instance();
    py::print(s);
}

} // anonymous namespace

OprChecker::OprChecker(std::shared_ptr<OpDef> opdef)
    : m_op(opdef) {}

void OprChecker::run(std::vector<InputSpec> inp_keys) {
    HostTensorGenerator<> gen;
    size_t nr_inps = inp_keys.size();
    SmallVector<HostTensorND> host_inp(nr_inps);
    VarNodeArray sym_inp(nr_inps);
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    for (size_t i = 0; i < nr_inps; ++ i) {
        //TODO: remove std::visit for support osx 10.12
        host_inp[i] = std::visit([&gen](auto&& arg) -> HostTensorND {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<TensorShape, T>) {
                    return *gen(arg);
                } else {
                    static_assert(std::is_same_v<HostTensorND, T>);
                    return arg;
                }
            }, inp_keys[i]);
        sym_inp[i] = opr::SharedDeviceTensor::make(*graph, host_inp[i]).node();
    }
    auto sym_oup = OpDef::apply_on_var_node(*m_op, sym_inp);
    size_t nr_oups = sym_oup.size();
    ComputingGraph::OutputSpec oup_spec(nr_oups);
    SmallVector<HostTensorND> host_sym_oup(nr_oups);
    for (size_t i = 0; i < nr_oups; ++ i) {
        oup_spec[i] = make_callback_copy(sym_oup[i], host_sym_oup[i]);
    }
    auto func = graph->compile(oup_spec);

    SmallVector<TensorPtr> imp_physical_inp(nr_inps);
    for (size_t i = 0; i < nr_inps; ++ i) {
        imp_physical_inp[i] = Tensor::make(host_inp[i]);
    }

    auto imp_oup = OpDef::apply_on_physical_tensor(*m_op, imp_physical_inp);
    mgb_assert(imp_oup.size() == nr_oups);

    // check input not modified
    for (size_t i = 0; i < imp_physical_inp.size(); ++i) {
        HostTensorND hv;
        hv.copy_from(imp_physical_inp[i]->dev_tensor()).sync();
        MGB_ASSERT_TENSOR_EQ(hv, host_inp[i]);
    }

    SmallVector<HostTensorND> host_imp_oup(nr_oups);
    for (size_t i = 0; i < nr_oups; ++ i) {
        host_imp_oup[i].copy_from(imp_oup[i]->dev_tensor()).sync();
    }

    func->execute().wait(); // run last because it may contain inplace operations

    for(size_t i = 0; i < nr_oups; ++ i) {
        MGB_ASSERT_TENSOR_EQ(host_sym_oup[i], host_imp_oup[i]);
    }
}

TEST(TestHelper, PyModule) {
    py::module m = PyEnv::get();
    py::print(m);
    py::print(py::cast(DeviceTensorND()));
}

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
