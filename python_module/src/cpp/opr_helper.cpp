/**
 * \file python_module/src/cpp/opr_helper.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./opr_helper.h"
#include "./megbrain_wrap.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/opr_load_dump.h"

using namespace mgb;

namespace {
    class OprParamsLoadContext final: public serialization::OprLoadContextRawPOD {
        PyObject *m_params;
        ComputingGraph *m_graph;
        size_t m_nr_used_params = 0, m_param_size = 0;
        size_t m_item_bytes_consumed = 0;

        void read_raw(void *dest, size_t size) override final {
            mgb_assert(m_nr_used_params < m_param_size);
            auto item = PyList_GetItem(m_params, m_nr_used_params);
            mgb_assert(item, "failed to get item %zu", m_nr_used_params);
            mgb_assert(PyBytes_Check(item), "list item must be bytes");
            auto item_size = PyBytes_Size(item);
            mgb_assert(size < (SIZE_MAX >> 3));
            mgb_assert(m_item_bytes_consumed + size <= size_t(item_size));
            auto item_buf = PyBytes_AsString(item);
            mgb_assert(item_size > 0 && item_buf);
            memcpy(dest, item_buf + m_item_bytes_consumed, size);
            m_item_bytes_consumed += size;
            if (m_item_bytes_consumed == size_t(item_size)) {
                ++ m_nr_used_params;
                m_item_bytes_consumed = 0;
            }
        }

        std::shared_ptr<HostTensorND> load_tensor() override {
            mgb_assert(0);
        }

        std::shared_ptr<DeviceTensorND> load_tensor_shared() override {
            mgb_assert(0);
        }

        const serialization::GraphLoadConfig& config() const override {
            mgb_assert(0);
        }

        public:
            OprParamsLoadContext(PyObject *params, ComputingGraph *graph):
                m_params{params}, m_graph{graph}
            {
                mgb_assert(PyList_Check(params), "params must be a list");
                m_param_size = PyList_Size(params);
            }

            ~OprParamsLoadContext() {
                mgb_assert(m_nr_used_params == m_param_size,
                        "number of params mismatch");
            }

            ComputingGraph& graph() override {
                return *m_graph;
            }
    };
} // anonymous namespace

_SplitPartCallback::callback_t _SplitPartCallback::make_callback() {
    mgb_assert(!m_cb_created);
    m_cb_created = true;

    std::shared_ptr<_SplitPartCallback> cb_ptr(this);

    auto cb = [cb_ptr](size_t sz) {
        return cb_ptr->call(sz);
    };

    return cb;
}

_SetGradCallback::callback_t _SetGradCallback::make_callback() {
    mgb_assert(!m_cb_created);
    m_cb_created = true;

    if (empty()) {
        return {};
    }

    std::shared_ptr<_SetGradCallback> cb_ptr(this);

    auto cb = [cb_ptr](const opr::SetGrad& opr) {
        auto graph = CompGraph::make_from_weak_ptr(
                opr.owner_graph()->shared_from_this());
        return cb_ptr->call(graph);
    };

    return cb;
}

_TimeoutCallback::callback_t _TimeoutCallback::make_callback() {
    mgb_assert(!m_cb_created);
    m_cb_created = true;

    std::shared_ptr<_TimeoutCallback> cb_ptr(this);
    auto cb = [cb_ptr]() {
        return cb_ptr->call();
    };
    return cb;
}

mgb::SymbolVar _create_subtensor_like_opr(
        const std::string &name,
        const SymbolVarArray& inputs,
        const std::vector<AxisIndexer> &idx,
        const mgb::OperatorNodeConfig &config) {
#define CHK1(_name, _opr) \
    if (name == _name) { \
        mgb_assert(inputs.size() == 1); \
        return opr::_opr::make(inputs[0], idx, config); \
    }
#define CHK2(_name, _opr) \
    if (name == _name) { \
        mgb_assert(inputs.size() == 2); \
        return opr::_opr::make(inputs[0], inputs[1], idx, config); \
    }

    CHK1("subtensor", Subtensor);
    CHK2("set_subtensor", SetSubtensor);
    CHK2("incr_subtensor", IncrSubtensor);
    CHK1("mavi", IndexingMultiAxisVec);
    CHK2("set_mavi", IndexingSetMultiAxisVec);
    CHK2("incr_mavi", IndexingIncrMultiAxisVec);
    CHK1("mesh_indexing", MeshIndexing);
    CHK1("batched_mesh_indexing", BatchedMeshIndexing);
    CHK2("incr_mesh_indexing", IncrMeshIndexing);
    CHK2("set_mesh_indexing", SetMeshIndexing);
    CHK2("batched_incr_mesh_indexing", BatchedIncrMeshIndexing);
    CHK2("batched_set_mesh_indexing", BatchedSetMeshIndexing);

    mgb_throw(MegBrainError, "bad subtensor opr name: %s", name.c_str());

#undef CHK1
#undef CHK2
}

SymbolVar _make_immutable(CompGraph &comp_graph, PyObject *npyarr,
        PyObject *dtype, const mgb::cg::OperatorNodeConfig &config) {

    auto cn = config.get_single_comp_node();
    mgb_assert(cn.valid(), "invalid comp node given to make_tensor");
    DType dtype_mgb;
    if (dtype && dtype != Py_None)
        dtype_mgb = npy::dtype_np2mgb(dtype);
    auto hv = npy::np2tensor(npyarr, npy::Meth::borrow(cn), dtype_mgb);
    return opr::ImmutableTensor::make(comp_graph.get(), hv, config);
}

SymbolVarArray _create_opr(
        const char *name, const SymbolVarArray &inputs,
        PyObject *params, const OperatorNodeConfig &config) {
    mgb_assert(!inputs.empty());
    auto registry = serialization::OprRegistry::find_by_name(name);
    mgb_assert(registry, "operator %s not found", name);
    OprParamsLoadContext ctx{params, inputs[0].node()->owner_graph()};
    VarNodeArray vinputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++ i)
        vinputs[i] = inputs[i].node();
    auto opr = registry->loader(ctx, vinputs, config);

    SymbolVarArray ret;
    for (auto i: opr->output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT))
            ret.push_back(i);
    }
    return ret;
}

#if MGB_ENABLE_OPR_MM
mgb::opr::CollectiveComm::Param load_collective_comm_params(
        PyObject* params, mgb::ComputingGraph* graph) {
    OprParamsLoadContext ctx{params, graph};
    return ctx.read_param<mgb::opr::CollectiveComm::Param>();
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
