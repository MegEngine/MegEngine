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
#include "megbrain/imperative/ops/elemwise.h"
#include "megbrain/imperative/ops/batch_norm.h"
#include "megbrain/imperative/ops/broadcast.h"
#include "megbrain/imperative/ops/utility.h"

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

    py::class_<VirtualDep, std::shared_ptr<VirtualDep>, OpDef>(m, "VirtualDep")
        .def(py::init<>());

    py::class_<CondTake, std::shared_ptr<CondTake>, OpDef>(m, "CondTake")
        .def(py::init<>());

    py::class_<NMSKeep, std::shared_ptr<NMSKeep>, OpDef>(m, "NMSKeep")
        .def(py::init<float, uint32_t>())
        .def_readwrite("iou_thresh", &NMSKeep::iou_thresh)
        .def_readwrite("max_output", &NMSKeep::max_output);

    py::class_<Elemwise, std::shared_ptr<Elemwise>, OpDef> elemwise(m, "Elemwise");
        elemwise.def(py::init<Elemwise::Mode>())
                .def_readwrite("mode", &Elemwise::mode);

#define V(m) .value(#m, Elemwise::Mode::m)
    py::enum_<Elemwise::Mode>(elemwise, "Mode")
        V(RELU)
        V(ABS)
        V(ACOS)
        V(ASIN)
        V(CEIL)
        V(COS)
        V(EXP)
        V(EXPM1)
        V(FLOOR)
        V(LOG)
        V(LOG1P)
        V(NEGATE)
        V(SIGMOID)
        V(SIN)
        V(TANH)
        V(ABS_GRAD)
        V(ADD)
        V(FLOOR_DIV)
        V(MAX)
        V(MIN)
        V(MOD)
        V(MUL)
        V(POW)
        V(SIGMOID_GRAD)
        V(SUB)
        V(SWITCH_GT0)
        V(TANH_GRAD)
        V(TRUE_DIV)
        V(LOG_SUM_EXP)
        V(LT)
        V(LEQ)
        V(EQ)
        V(SHL)
        V(SHR)
        V(COND_LEQ_MOV)
        V(FUSE_MUL_ADD3)
        V(FUSE_MUL_ADD4)
        V(FUSE_ADD_RELU)
        V(FUSE_ADD_SIGMOID)
        V(FUSE_ADD_TANH)
        V(FAST_TANH)
        V(FAST_TANH_GRAD)
        V(ROUND)
        V(RMULH)
        V(ATAN2)
        V(ERF)
        V(ERFINV)
        V(ERFC)
        V(ERFCINV)
        V(H_SWISH)
        V(H_SWISH_GRAD)
        V(FUSE_ADD_H_SWISH)
        V(NOT)
        V(AND)
        V(OR)
        V(XOR);
#undef V

    py::class_<BatchNorm, std::shared_ptr<BatchNorm>, OpDef> batchnorm(m, "BatchNorm");
        batchnorm.def(py::init<const BatchNorm::Param::ParamDim&, const BatchNorm::Param::FwdMode&, double, double, float, float>())
                 .def_readwrite("param_dim", &BatchNorm::param_dim)
                 .def_readwrite("fwd_mode", &BatchNorm::fwd_mode)
                 .def_readwrite("epsilon", &BatchNorm::epsilon)
                 .def_readwrite("avg_factor", &BatchNorm::avg_factor)
                 .def_readwrite("scale", &BatchNorm::scale)
                 .def_readwrite("bias", &BatchNorm::bias);

#define V(m) .value(#m, BatchNorm::Param::ParamDim::m)
    py::enum_<BatchNorm::Param::ParamDim>(batchnorm, "ParamDim")
        V(DIM_11HW)
        V(DIM_1CHW)
        V(DIM_1C11);
#undef V

#define V(m) .value(#m, BatchNorm::Param::FwdMode::m)
    py::enum_<BatchNorm::Param::FwdMode>(batchnorm, "FwdMode")
        V(TRAINING)
        V(INFERENCE);
#undef V

    py::class_<Broadcast, std::shared_ptr<Broadcast>, OpDef>(m, "Broadcast")
        .def(py::init<>());

}
