/**
 * \file imperative/python/src/ops.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./ops.h"

#include "megbrain/imperative.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/tensor_manip.h"
#include "megbrain/imperative/ops/collective_comm.h"
#include "megbrain/imperative/ops/io_remote.h"
#include "megbrain/imperative/ops/cond_take.h"
#include "megbrain/imperative/ops/nms.h"

namespace py = pybind11;

void init_ops(py::module m) {
    using namespace mgb::imperative;

    py::class_<OprAttr, std::shared_ptr<OprAttr>, OpDef>(m, "OprAttr")
        .def(py::init<>())
        .def_readwrite("type", &OprAttr::type)
        .def_readwrite("param", &OprAttr::param)
        .def_readwrite("config", &OprAttr::config)
        .def_property("param",
            [](const OprAttr& attr) -> py::bytes {
                return std::string(attr.param.begin(), attr.param.end());
            },
            [] (OprAttr& attr, py::bytes data) {
                auto s = py::cast<std::string>(data);
                attr.param.clear();
                attr.param.insert(attr.param.end(), s.begin(), s.end());
            });

    py::class_<BackwardGraph, std::shared_ptr<BackwardGraph>, OpDef>(m, "BackwardGraph")
        .def("interpret", [](BackwardGraph& self, py::object pyf, py::object pyc,
                             const mgb::SmallVector<py::object>& inputs) {
                auto f = [pyf](OpDef& op, const mgb::SmallVector<py::object>& inputs) {
                    return py::cast<mgb::SmallVector<py::object>>(pyf(op.copy(), inputs));
                };
                auto c = [pyc](const TensorPtr& tensor) {
                    return pyc(tensor->dev_tensor());
                };
                return self.graph().interpret<py::object>(f, c, inputs);
            });

    py::class_<GetVarShape, std::shared_ptr<GetVarShape>, OpDef>(m, "GetVarShape")
        .def(py::init());

#define V(m) .value(#m, CollectiveComm::Mode::m)
    py::enum_<CollectiveComm::Mode>(m, "CollectiveCommMode")
        V(REDUCE_SUM)
        V(BROADCAST)
        V(ALL_GATHER)
        V(REDUCE_SCATTER_SUM)
        V(ALL_REDUCE_SUM)
        V(ALL_REDUCE_MAX)
        V(ALL_REDUCE_MIN)
        V(ALL_REDUCE_PROD)
        V(GATHER)
        V(SCATTER)
        V(ALL_TO_ALL);
#undef V

    py::class_<CollectiveComm, std::shared_ptr<CollectiveComm>, OpDef>(m, "CollectiveComm")
        .def(py::init<>())
        .def_readwrite("key", &CollectiveComm::key)
        .def_readwrite("nr_devices", &CollectiveComm::nr_devices)
        .def_readwrite("rank", &CollectiveComm::rank)
        .def_readwrite("is_root", &CollectiveComm::is_root)
        .def_readwrite("local_grad", &CollectiveComm::local_grad)
        .def_readwrite("addr", &CollectiveComm::addr)
        .def_readwrite("port", &CollectiveComm::port)
        .def_readwrite("mode", &CollectiveComm::mode)
        .def_readwrite("dtype", &CollectiveComm::dtype)
        .def_readwrite("backend", &CollectiveComm::backend)
        .def_readwrite("comp_node", &CollectiveComm::comp_node);

    py::class_<RemoteSend, std::shared_ptr<RemoteSend>, OpDef>(m, "RemoteSend")
        .def(py::init<>())
        .def_readwrite("key", &RemoteSend::key)
        .def_readwrite("addr", &RemoteSend::addr)
        .def_readwrite("port", &RemoteSend::port)
        .def_readwrite("rank_to", &RemoteSend::rank_to);

    py::class_<RemoteRecv, std::shared_ptr<RemoteRecv>, OpDef>(m, "RemoteRecv")
        .def(py::init<>())
        .def_readwrite("key", &RemoteRecv::key)
        .def_readwrite("addr", &RemoteRecv::addr)
        .def_readwrite("port", &RemoteRecv::port)
        .def_readwrite("rank_from", &RemoteRecv::rank_from)
        .def_readwrite("shape", &RemoteRecv::shape)
        .def_readwrite("cn", &RemoteRecv::cn)
        .def_readwrite("dtype", &RemoteRecv::dtype);

    py::class_<ParamPackSplit, std::shared_ptr<ParamPackSplit>, OpDef>(m, "ParamPackSplit")
        .def(py::init<>())
        .def_readwrite("offsets", &ParamPackSplit::offsets)
        .def_readwrite("shapes", &ParamPackSplit::shapes);

    py::class_<ParamPackConcat, std::shared_ptr<ParamPackConcat>, OpDef>(m, "ParamPackConcat")
        .def(py::init<>())
        .def_readwrite("offsets", &ParamPackConcat::offsets);

    py::class_<CondTake, std::shared_ptr<CondTake>, OpDef>(m, "CondTake")
        .def(py::init<>());

    py::class_<NMSKeep, std::shared_ptr<NMSKeep>, OpDef>(m, "NMSKeep")
        .def(py::init<float, uint32_t>())
        .def_readwrite("iou_thresh", &NMSKeep::iou_thresh)
        .def_readwrite("max_output", &NMSKeep::max_output);
}
