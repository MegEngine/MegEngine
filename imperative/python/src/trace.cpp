/**
 * \file imperative/python/src/trace.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./trace.h"
#include "./helper.h"
#include "megbrain/imperative/ops/autogen.h"

namespace py = pybind11;

namespace mgb::imperative::python {

apply_result_t apply_trace(ApplyContext& ctx) {
    apply_result_t outputs;

    if (ctx.backward) {
        // reach here when compiled=True
        // call megbrain_graph.py apply(BackwardGraph, *args)
        auto args = py::tuple(ctx.nargs + 1);
        args[0] = py::cast(ctx.op);
        for (size_t i = 0; i < ctx.nargs; i++) {
            args[i + 1] = py::cast(ctx.args[i]->m_var);
        }
        py::object ret = py::reinterpret_steal<py::object>(
                PyObject_Call(cpp_apply_backward_varnode, args.ptr(), nullptr));
        if (!ret) {
            throw py::value_error("invalid py object call");
        }

        // assumption: python function always returns PyList
        auto tup = py::reinterpret_borrow<py::list>(ret);
        for (auto i = 0; i < tup.size(); i++) {
            auto pitem = tup[i].cast<cg::VarNode*>();
            outputs.emplace_back(std::make_shared<Tensor>(pitem));
        }
        return outputs;
    }

    PyObject* pyf;
    if (is_compiled) {
        // run apply in compiled mode, step 2, 3, etc
        pyf = cpp_apply_compiled_mode;
    } else {
        // run first step, both symbolic and non symbolic
        pyf = cpp_apply_with_tracing;
    }

    auto args = py::tuple(ctx.nargs + 1);
    args[0] = py::cast(ctx.op);
    for (size_t i = 0; i < ctx.nargs; i++) {
        args[i + 1] = TensorWrapper::make(ctx.args[i]->shared_from_this());
    }
    auto ret = py::reinterpret_steal<py::object>(
            PyObject_Call(pyf, args.ptr(), nullptr));

    // assumption: python function always returns PyList
    auto tup = py::reinterpret_borrow<py::list>(ret);
    for (auto i = 0; i < tup.size(); i++) {
        auto tw = TensorWrapper::try_cast(tup[i].ptr());
        outputs.emplace_back(tw->m_tensor);
    }
    return outputs;
}

} // namespace mgb::imperative::python
