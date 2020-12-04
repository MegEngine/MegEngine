/**
 * \file imperative/python/src/imperative_rt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./imperative_rt.h"

#include <future>
#include <variant>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "megbrain/imperative.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "./helper.h"
#include "./common.h"

namespace py = pybind11;

using namespace mgb;
using namespace imperative;
using namespace interpreter;


namespace {

std::optional<std::tuple<std::shared_ptr<OpDef>, std::vector<bool>, std::vector<bool>>>
make_backward_graph(
    const OpDef& opdef, std::vector<LogicalTensorDesc> inputs,
    std::vector<bool> input_requires_grad,
    std::vector<bool> output_has_grad) {
    auto res = OpDef::make_backward_graph(opdef,
        SmallVector<LogicalTensorDesc>(inputs.begin(), inputs.end()),
        SmallVector<bool>(input_requires_grad.begin(), input_requires_grad.end()),
        SmallVector<bool>(output_has_grad.begin(), output_has_grad.end()));
    if (res.backward) {
        return std::optional<std::tuple<std::shared_ptr<OpDef>, std::vector<bool>, std::vector<bool>>>{
                std::in_place, res.backward, res.save_for_backward, res.input_has_grad};
    } else {
        return {};
    }
}
} // namespace

void init_imperative_rt(py::module m) {
    py::class_<Interpreter::Channel>(m, "Interpreter")
        .def("put", [](Interpreter::Channel& self, py::array data, DType dtype, CompNode cn) {
                if (!cn.valid()) {
                    cn = CompNode::load(get_default_device());
                }
                constexpr int size_threshhold = TensorShape::MAX_NDIM;
                if (data.size() > size_threshhold) {
                    return self.put(npy::np2tensor(data.ptr(), npy::Meth::borrow(cn), dtype));
                } else {
                    HostTensorND ret(cn);
                    return self.put(npy::np2tensor(data.ptr(), npy::Meth::copy_into(&ret), dtype));
                }
            }, py::arg(), py::arg("dtype") = py::none(), py::arg("device") = py::none())
        .def("put", py::overload_cast<const DeviceTensorND&>(&Interpreter::Channel::put))
        .def("delete", [](Interpreter::Channel& self, Interpreter::Handle handle) {
                return self.del(handle);
            })
        .def("_swap_in", [](Interpreter::Channel& self, Interpreter::Handle handle) {
                self.swap_in(handle);
            })
        .def("_swap_out", [](Interpreter::Channel& self, Interpreter::Handle handle) {
                self.swap_out(handle);
            })
        .def("_drop", [](Interpreter::Channel& self, Interpreter::Handle handle) {
                self.drop(handle);
            })
        .def("get_value", [](Interpreter::Channel& self, Interpreter::Handle handle) {
                PyObject* optr = npy::ndarray_from_tensor(self.get_value(handle), npy::ShareType::TRY_SHARE);
                return py::reinterpret_steal<py::object>(optr);
            })
        .def("get_dtype", &Interpreter::Channel::get_dtype)
        .def("get_device", &Interpreter::Channel::get_device)
        .def("get_shape", &Interpreter::Channel::get_shape)
        .def("_get_dev_tensor", &Interpreter::Channel::get_dev_tensor)
        .def("_set_swap_flag", &Interpreter::Channel::set_swap_flag)
        .def("_set_drop_flag", &Interpreter::Channel::set_drop_flag)
        .def("apply_op", &Interpreter::Channel::apply_op)
        .def("config_async_level", &Interpreter::Channel::config_async_level)
        .def("get_async_level", &Interpreter::Channel::get_async_level)
        .def("sync", &Interpreter::Channel::sync, py::call_guard<py::gil_scoped_release>());

    std::unique_ptr<Interpreter::Channel> ch = Interpreter::inst().create_channel();
    m.attr("interpreter") = py::detail::make_caster<decltype(ch)>::cast(
        std::move(ch), py::return_value_policy::move, {});
    for (auto name : {"put", "delete", "get_value", "get_dtype", "get_device", "get_shape", "_get_dev_tensor", "apply_op", "config_async_level", "get_async_level", "_drop", "_swap_in", "_swap_out", "_set_drop_flag", "_set_swap_flag"}) {
        m.attr(name) = m.attr("interpreter").attr(name);
    }

    m.def("sync", [m]() {
            m.attr("interpreter").attr("sync")();
            py::gil_scoped_release _;
            py_task_q.wait_all_task_finish();
         });

    m.def("make_backward_graph", &make_backward_graph);
}
