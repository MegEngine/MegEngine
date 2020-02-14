/**
 * \file python_module/src/cpp/opr_defs.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief extra opr definitions
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#ifndef SWIG
#pragma once

#include "./megbrain_wrap.h"
#include "./opr_helper.h"

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/collective_comm.h"
#endif
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/tensor_manip.h"
using mgb::SymbolVar;
using mgb::SymbolVarArray;
using mgb::OperatorNodeConfig;

#endif

class _Opr {

public:

// basic arith

static SymbolVar add_update(SymbolVar dest, SymbolVar delta,
        const SharedScalar &alpha, const SharedScalar &beta,
        const SharedScalar &bias, const SharedScalar &disable,
        const OperatorNodeConfig &config) {
    return mgb::opr::AddUpdate::make(dest, delta,
            {alpha.get_val(), beta.get_val(), bias.get_val(), disable.get_val()},
            config);
}

// tensor manip

static SymbolVarArray param_pack_split(
        SymbolVar src, SymbolVar table,
        const std::vector<std::vector<size_t>>& shapes,
        const OperatorNodeConfig& config);

static SymbolVar dimshuffle(SymbolVar src,
        const std::vector<int> &pattern, size_t ndim,
        const OperatorNodeConfig &config) {
    return mgb::opr::Dimshuffle::make(src, pattern, ndim, config);
}

static SymbolVar _axis_add_remove(SymbolVar src,
        const std::vector<int>& axis, bool is_add,
        const OperatorNodeConfig &config);

static SymbolVar callback_injector(SymbolVar src, _CompGraphCallback &callback,
        const OperatorNodeConfig &config) {
    return mgb::opr::CallbackInjector::make(src, callback.make_callback());
}

static SymbolVar callback_injector(SymbolVarArray src, _CompGraphCallback &callback,
                                   const OperatorNodeConfig &config) {
    return mgb::opr::CallbackInjector::make(src, callback.make_multi_input_callback());
}

static SymbolVar set_grad(SymbolVar src, _SetGradCallback &grad_getter,
        const OperatorNodeConfig &config) {
    return mgb::opr::SetGrad::make(src, grad_getter.make_callback(), config);
}

// multi machine

static SymbolVar lock_acquire(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config);

static SymbolVar lock_release(SymbolVar var, size_t lock_id, size_t group_id,
        const OperatorNodeConfig &config);

static SymbolVar remote_send(
        const std::string& server_addr, const int port,
        const std::string& key, SymbolVar var,
        const bool is_grad,
        const OperatorNodeConfig& config);

static SymbolVar remote_recv(const std::string& server_addr, const int port,
                             const std::string& key,
                             CompGraph& graph,
                             const std::vector<size_t>& shape, PyObject* dtype,
                             const OperatorNodeConfig& config);

static SymbolVar collective_comm_with_input(
        SymbolVar inpvar, const std::string& key, const size_t nr_devices,
        const uint32_t rank, const uint32_t root, const std::string& server_addr,
        const int port, PyObject* params, PyObject* dtype,
        const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable);

static SymbolVar collective_comm_without_input(
        CompGraph& graph, const std::string& key, const size_t nr_devices,
        const uint32_t rank, const uint32_t root, const std::string& server_addr,
        const int port, PyObject* params, PyObject* dtype,
        const std::string& backend, SharedND* output_buf,
        const OperatorNodeConfig& config, const SharedScalar& disable);

// misc
static SymbolVarArray extern_c_opr_placeholder(
        const SymbolVarArray& inputs,
        const std::vector<std::vector<size_t>>& output_shapes,
        PyObject* dtypes,
        const char* dump_name, PyObject* data_bytes,
        const OperatorNodeConfig& config);

static SymbolVarArray tensor_rt_runtime(const SymbolVarArray& inputs,
                                        PyObject* data_bytes,
                                        const OperatorNodeConfig& config);

static SymbolVar timestamp(SymbolVar input, PyObject* dest, size_t dest_off,
                           const OperatorNodeConfig& config);

static SymbolVar virtual_loss(const SymbolVarArray& ys,
                              const SymbolVarArray& y_grads,
                              const OperatorNodeConfig& config);

static SymbolVar virtual_dep(const SymbolVarArray& symvars,
                             const OperatorNodeConfig& config);


#ifdef SWIG
%pythoncode {

@classmethod
def _make_axis_vec(cls, axis):
    ret = _VectorInt()
    if isinstance(axis, collections.Iterable):
        for i in axis:
            ret.push_back(i)
    else:
        ret.push_back(axis)
    return ret

@classmethod
def add_axis(cls, src, axis, config):
    return cls._axis_add_remove(src, cls._make_axis_vec(axis), True, config)

@classmethod
def remove_axis(cls, src, axis, config):
    return cls._axis_add_remove(src, cls._make_axis_vec(axis), False, config)

} // %pythoncode
#endif // SWIG

};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
