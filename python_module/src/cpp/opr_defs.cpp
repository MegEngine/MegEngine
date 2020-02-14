/**
 * \file python_module/src/cpp/opr_defs.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./opr_defs.h"
#include "./opr_helper.h"
#include "./python_helper.h"

#if MGB_ENABLE_OPR_MM
#include "mm_handler.h"
#endif

#include "megbrain/opr/io.h"
#include "megbrain/serialization/extern_c_opr_io.h"

using namespace mgb;
using namespace mgb::opr;

SymbolVar _Opr::_axis_add_remove(SymbolVar src,
        const std::vector<int>& axis, bool is_add,
        const OperatorNodeConfig &config) {
    using ADR = mgb::opr::AxisAddRemove;
    std::vector<ADR::AxisDesc> desc;
    mgb_assert(!axis.empty());
    for (auto i: axis) {
        if (is_add) {
            desc.emplace_back(ADR::AxisDesc::make_add(i));
        } else {
            desc.emplace_back(ADR::AxisDesc::make_remove(i));
        }
    }
    return ADR::make(src, desc, config);
}

SymbolVarArray _Opr::param_pack_split(
        SymbolVar src, SymbolVar table,
        const std::vector<std::vector<size_t>>& shapes,
        const OperatorNodeConfig& config) {
    auto size = shapes.size();
    mgb::TensorShapeArray shapearr(size);
    for (size_t i = 0; i < size; i++) {
        shapearr[i] = npy::vec2shape(shapes[i]);
    }

    if (!table.node()) {
        auto cn = src.node()->comp_node();
        if (config.has_comp_node_set()) {
            cn = config.get_single_comp_node();
        }
        auto table_val = megdnn::ParamPackSplit::gen_table(
                shapearr, cn.get_mem_addr_alignment(), src.dtype().size());
        HostTensorND hv{cn, TensorShape{table_val.size()}, dtype::Int32{}};
        memcpy(hv.raw_ptr(), table_val.data(), table_val.size() * sizeof(int));
        table = opr::ImmutableTensor::make(*src.node()->owner_graph(), hv);
    }

    return mgb::opr::ParamPackSplit::make(src, table, shapearr, config);
}

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/lock.h"
#include "megbrain/opr/io_remote.h"

SymbolVar _Opr::lock_acquire(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config) {
    return mgb::opr::LockAcquire::make(var, {lock_id, group_id}, config);
}

SymbolVar _Opr::lock_release(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config) {
    return mgb::opr::LockRelease::make(var, {lock_id, group_id}, config);
}

SymbolVar _Opr::remote_send(
        const std::string& server_addr, const int port,
        const std::string& key, SymbolVar var,
        const bool is_grad,
        const OperatorNodeConfig& config) {
    return RemoteSend::make({key, RemoteIOBase::Type::SEND, is_grad}, var,
                            std::make_shared<GroupClientProxy>(ssprintf(
                                    "%s:%d", server_addr.c_str(), port)),
                            config);
}

SymbolVar _Opr::remote_recv(const std::string& server_addr, const int port,
                            const std::string& key, CompGraph& graph,
                            const std::vector<size_t>& shape, PyObject* dtype,
                            const OperatorNodeConfig& config) {
    const TensorShape ishape = npy::vec2shape(shape);
    const DType idtype = npy::dtype_np2mgb(dtype);

    return RemoteRecv::make({key, RemoteIOBase::Type::RECV, false},
                            graph.get(),
                            std::make_shared<GroupClientProxy>(
                                    ssprintf("%s:%d", server_addr.c_str(), port)),
                            config, ishape, idtype);
}

SymbolVar _Opr::collective_comm_with_input(
        SymbolVar inpvar, const std::string& key,
        const size_t nr_devices, const uint32_t rank, const uint32_t root,
        const std::string& server_addr, const int port,
        PyObject* params, PyObject* dtype,
        const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable) {
    SymbolVarArray inputs(1, inpvar);
    ComputingGraph* graph = inpvar.node()->owner_graph();
    auto group_mgr = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", server_addr.c_str(), port));
    SmallVector<std::shared_ptr<mgb::DeviceTensorND>> dev_buffer_arr(1, nullptr);
    if (output_buf)
        dev_buffer_arr[0] = output_buf->dev_tensor();
    CollectiveComm::Param param = load_collective_comm_params(params, graph);
    mgb::DType _dtype = DType();
    if (dtype != Py_None) {
        _dtype = npy::dtype_np2mgb(dtype);
    }
    return CollectiveComm::make(inputs, graph, key, nr_devices, rank, root, group_mgr,
            dev_buffer_arr, param, _dtype, backend, config, disable.get_val())[0];
}

SymbolVar _Opr::collective_comm_without_input(
        CompGraph& cg, const std::string& key,
        const size_t nr_devices, const uint32_t rank, const uint32_t root,
        const std::string& server_addr, const int port,
        PyObject* params, PyObject* dtype,
        const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable) {
    SymbolVarArray inputs;
    auto& graph = cg.get();
    auto group_mgr = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", server_addr.c_str(), port));
    SmallVector<std::shared_ptr<mgb::DeviceTensorND>> dev_buffer_arr(1, nullptr);
    if (output_buf)
        dev_buffer_arr[0] = output_buf->dev_tensor();
    CollectiveComm::Param param = load_collective_comm_params(params, &graph);
    mgb::DType _dtype = DType();
    if (dtype != Py_None) {
        _dtype = npy::dtype_np2mgb(dtype);
    }
    return CollectiveComm::make(inputs, &graph, key, nr_devices, rank, root, group_mgr,
            dev_buffer_arr, param, _dtype, backend, config, disable.get_val())[0];
}

#else
namespace {
    [[noreturn]] void on_opr_mm() {
        mgb_throw(MegBrainError, "opr-mm disabled at compile time");
    }
}
SymbolVar _Opr::lock_acquire(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config) {
    on_opr_mm();
}

SymbolVar _Opr::lock_release(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config) {
    on_opr_mm();
}


SymbolVar _Opr::remote_send(
        const std::string& server_addr, const int port,
        const std::string& key, SymbolVar var,
        const bool is_grad,
        const OperatorNodeConfig& config) {
    on_opr_mm();
}

SymbolVar _Opr::remote_recv(const std::string& server_addr, const int port,
                            const std::string& key, CompGraph& graph,
                            const std::vector<size_t>& shape, PyObject* dtype,
                            const OperatorNodeConfig& config) {
    on_opr_mm();
}

SymbolVar _Opr::collective_comm_with_input(
        SymbolVar inpvar, const std::string& key,
        const size_t nr_devices, const uint32_t rank, const uint32_t root,
        const std::string& server_addr, const int port, PyObject* params,
        PyObject* dtype, const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable) {
    on_opr_mm();
}

SymbolVar _Opr::collective_comm_without_input(
        CompGraph& cg, const std::string& key,
        const size_t nr_devices, const uint32_t rank, const uint32_t root,
        const std::string& server_addr, const int port, PyObject* params,
        PyObject* dtype, const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable) {
    on_opr_mm();
}

#endif // MGB_ENABLE_OPR_MM

SymbolVarArray _Opr::extern_c_opr_placeholder(
        const SymbolVarArray& inputs,
        const std::vector<std::vector<size_t>>& output_shapes,
        PyObject* output_dtypes, const char* dump_name, PyObject* data_bytes,
        const OperatorNodeConfig& config) {
    mgb_assert(PyBytes_Check(data_bytes));
    if (output_dtypes != Py_None) {
        mgb_assert(PyTuple_Check(output_dtypes));
        mgb_assert(output_shapes.size() ==
                           static_cast<size_t>(PyTuple_Size(output_dtypes)));
    }

    TensorShapeArray cpp_output_shapes(output_shapes.size());
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        cpp_output_shapes[i] = npy::vec2shape(output_shapes[i]);
    }
    SmallVector<DType> cpp_output_dtypes;
    if (output_dtypes != Py_None) {
        size_t dtype_size = PyTuple_Size(output_dtypes);
        for (size_t i = 0; i < dtype_size; ++i) {
            cpp_output_dtypes.push_back(
                    npy::dtype_np2mgb(PyTuple_GetItem(output_dtypes, i)));
        }
    }

    auto opr = serialization::ExternCOprRunner::make_placeholder(
            inputs, cpp_output_shapes, dump_name, PyBytes_AsString(data_bytes),
            PyBytes_Size(data_bytes), config, cpp_output_dtypes);
    SymbolVarArray ret;
    ret.reserve(opr->output().size());
    for (auto i: opr->output())
        ret.emplace_back(i);
    return ret;
}

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_runtime_opr.h"

SymbolVarArray _Opr::tensor_rt_runtime(const SymbolVarArray& inputs,
                                       PyObject* data_bytes,
                                       const OperatorNodeConfig& config) {
    mgb_assert(PyBytes_Check(data_bytes));
    auto size = PyBytes_Size(data_bytes);
    mgb_assert(size, "trt data bytes should not be empty");
    return opr::TensorRTRuntimeOpr::make(PyBytes_AsString(data_bytes),
                                         size, inputs,
                                         config);
}
#else
SymbolVarArray _Opr::tensor_rt_runtime(const SymbolVarArray& inputs,
                                       PyObject* data_bytes,
                                       const OperatorNodeConfig& config) {
    mgb_throw(MegBrainError, "TensorRT disabled at compile time");
}
#endif

SymbolVar _Opr::timestamp(SymbolVar input, PyObject* dest, size_t dest_off,
                           const OperatorNodeConfig& config) {
    auto tensor = std::make_shared<HostTensorND>(
            npy::np2tensor(dest, npy::Meth::must_borrow(), dtype::Float32{}));
    return opr::Timestamp::make(input, std::move(tensor), dest_off, config);
}

SymbolVar _Opr::virtual_loss(const SymbolVarArray& ys,
                             const SymbolVarArray& y_grads,
                             const OperatorNodeConfig& config) {
    return opr::VirtualLoss::make(ys, y_grads, {}, config);
}

SymbolVar _Opr::virtual_dep(const SymbolVarArray& symvars,
                            const OperatorNodeConfig& config) {
    return opr::VirtualDep::make(symvars, config);
}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
