/**
 * \file python_module/src/cpp/megbrain_wrap.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief wrappers for basic functionalities
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "./python_helper.h"
#include "./megbrain_pubapi.h"

#include "megbrain/graph.h"
#include "megbrain/opr/io.h"

#include <map>
#include <string>

class CompGraph;
class CompGraphCallbackValueProxy;

/*!
 * \brief proxy a mgb::DeviceTensorND or a SymbolVar
 */
class SharedND {
    mgb::pubapi::DeviceTensor m_pubapi_dev_tensor;

    std::shared_ptr<mgb::DeviceTensorND> m_dev_tensor;
    mgb::HostTensorND m_async_copy_refkeeper;
    mgb::VarNode *m_var = nullptr;
    bool m_copy_sync = true;

    bool sync(mgb::DeviceTensorND &dv);
    inline void _check_before_share_memory(const SharedND& rhs);

    public:
        SharedND() = default;

        SharedND(mgb::CompNode node, PyObject* dtype):
            m_dev_tensor(std::make_shared<mgb::DeviceTensorND>(
                        node, npy::dtype_np2mgb(dtype)))
        { }

        SharedND(const std::shared_ptr<mgb::DeviceTensorND>& dv)
            : m_dev_tensor(dv) {}

        //! set init shape; can be only called once
        void _set_init_shape(const std::vector<size_t> &shape);

        //! resize to given shape
        void _resize(const std::vector<size_t> &shape);

	//! reset dev_tensor to zeros
	void _reset_zero();

        /*!
         * \brief assign to proxy given dev tensor; used by craniotome
         */
        void assign(const mgb::DeviceTensorND &dv) {
            mgb_assert(!m_dev_tensor && !m_var);
            m_dev_tensor = std::make_shared<mgb::DeviceTensorND>(dv);
        }

        /*!
         * \brief assign to proxy a var node; used by craniotome
         */
        void assign(mgb::VarNode *var) {
            mgb_assert(!m_dev_tensor && !m_var);
            m_var = var;
        }

        /*!
         * \brief share memory from another SharedND; only used in ParamPack
         */
        void _share_memory_from(const SharedND& rhs, size_t begin);

        /*!
        * \brief reset dev_tensor to another SharedNd's
        */
        void _reset_dev_tensor(const SharedND& rhs);

        uintptr_t _pubapi_dev_tensor_ptr(int version);

        mgb::SymbolVar _as_sym_var(CompGraph &cg, const std::string &name,
                bool volatile_);

        mgb::CompNode _get_comp_node() {
            return m_dev_tensor->comp_node();
        }

        void _set_copy_sync(bool flag) {
            m_copy_sync = flag;
        }

        //! get dev buffer from shared nd
        const std::shared_ptr<mgb::DeviceTensorND>& dev_tensor() {
            return m_dev_tensor;
        }

        void _copy_from_npyarr(PyObject *npyarr);
        void _copy_from_value_proxy(CompGraphCallbackValueProxy &value);
        void _share_from_value_proxy(CompGraphCallbackValueProxy &value);
        static SharedND _from_symvar(mgb::SymbolVar symvar);

        //! get numpy ndarray that contains a copy of the value; return new ref
        PyObject* _get_npyarr();
        PyObject* _get_dtype();
        std::vector<size_t> _get_shape();

        /*!
         * \brief copy to sub of this from another SharedND
         * \param axis axis for sub, or -1 to work on flattened array
         */
        void copy_to_sub_from_shared(
                int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step,
                const SharedND &rhs);

        /*!
         * \brief copy from sub of another SharedND to this
         * \param axis axis for sub, or -1 to work on flattened array, -2 to
         *      copy whole tensor, -3 to copy whole tensor fixlayout
         */
        void copy_from_shared_sub(const SharedND &rhs,
                int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step);
};

/*!
 * \brief wraps around shared pointer to mgb::HostTensorND
 */
class _HostSharedND {
    bool m_own_storage = false, m_borrow_on_cpu = false;
    std::shared_ptr<mgb::HostTensorND> m_tensor;
    //! set to non-null if this _HostSharedND is set to proxy a var
    mgb::opr::Host2DeviceCopy* m_proxied_opr = nullptr;

    void ensure_own_storage();

    public:
        _HostSharedND() = default;

        _HostSharedND(const _HostSharedND &rhs):
            m_own_storage{false},
            m_tensor{rhs.m_tensor},
            m_proxied_opr{rhs.m_proxied_opr}
        {
        }

        _HostSharedND(mgb::CompNode node, mgb::DType dtype):
            m_own_storage{true},
            m_tensor{std::make_shared<mgb::HostTensorND>(node, dtype)}
        {
        }

        _HostSharedND(mgb::CompNode node, PyObject* dtype):
            _HostSharedND(node, npy::dtype_np2mgb(dtype))
        {
        }

        _HostSharedND& operator = (const _HostSharedND &) = delete;

        /*!
         * \brief make a _HostSharedND by proxing a var produced by
         *      Host2DeviceCopy
         */
        static _HostSharedND make_proxy(mgb::SymbolVar var);

        mgb::SymbolVar _as_sym_var(CompGraph &cg, bool enable_static_infer,
                const std::string &name);

        void _resize(const std::vector<size_t> &shape);
        void _copy_from_npyarr(PyObject *npyarr, bool borrow);

        void _enable_borrow_on_cpu(bool flag) {
            m_borrow_on_cpu = flag;
        }

        std::string __repr__() const;
        PyObject* _get_dtype();
};

/*!
 * \brief proxy a value to be passed to computing graph callback
 */
class CompGraphCallbackValueProxy {
    mgb::pubapi::DeviceTensor m_pubapi_dev_tensor;
    bool m_is_active = false; //! setup called but on_finished not called
    bool m_use_raw_hv = false;
    bool m_value_used, m_eager_copy;
    mgb::HostTensorND m_hv;
    std::shared_ptr<mgb::CompNode::Event> m_copy_event;

    //! original dev value
    mgb::DeviceTensorND m_dev_value;

    //! perform D2H copy
    void do_copy();

    public:
        static CompGraphCallbackValueProxy make_raw_host_value_proxy(
                const mgb::HostTensorND &hv);

        bool eager_copy() const {
            return m_eager_copy;
        }

        mgb::DeviceTensorND& dev_tensor() {
            return m_dev_value;
        }

        void setup(const mgb::DeviceTensorND &val, bool eager_copy);
        void sync();

        /*!
         * \brief called after python callback returned
         */
        void on_finished();

        //! get numpy ndarray that contains a copy of the value; return new ref
        PyObject* _get_npyarr();
        PyObject* _get_dtype();
        std::vector<size_t> _get_shape();

        uintptr_t _pubapi_dev_tensor_ptr(int version);

        mgb::CompNode _get_comp_node();
};

class AsyncExec {
    public:
        class Core;

        AsyncExec() = default;

        ~AsyncExec();

        AsyncExec(std::unique_ptr<mgb::cg::AsyncExecutable> f);

        void _execute();
        void _wait();
        double _get_prev_exec_time();

        void clear_device_memory();

        std::vector<std::pair<mgb::CompNode, size_t>>
        _update_static_alloc_plan_and_get_size();

        std::string _to_json_str();

        /*!
         * \brief find all Host2DeviceCopy input vars that are mutable (i.e.
         *      used as func args)
         */
        mgb::SymbolVarArray _find_mutable_input();

        Core* core() const;

        void set_multi_part_par_graph(std::shared_ptr<mgb::ComputingGraph> g) {
            m_multi_part_par_graph = std::move(g);
        }

    private:
        std::shared_ptr<Core> m_core;
        //! parent graph in multi-part compiling
        std::shared_ptr<mgb::ComputingGraph> m_multi_part_par_graph;
};

/*!
 * \brief callback wrapper for computing graph
 */
class _CompGraphCallback {
    bool m_cb_created = false, m_eager_copy = false;
    AsyncExec::Core* m_ae_core = nullptr;
    std::vector<CompGraphCallbackValueProxy> m_value_proxies;

    public:
        /*!
         * \brief set AsyncExec associated with this callback; if it is set,
         *      eager value copy would be enabled
         */
        void set_async_exec(const AsyncExec &ae);

        /*!
         * \brief set whether enabling eager copy
         *
         * If eager copy is enabled, host to device copy would start immediately
         *      and asynchronously when this callback is executed by megbrain
         */
        void set_eager_copy(bool flag);

        virtual ~_CompGraphCallback() = default;

        std::function<void(mgb::SmallVector<mgb::DeviceTensorND> &)> make_multi_input_callback();
        std::function<void(mgb::DeviceTensorND &)> make_callback();

        /*!
         * \brief call python callback
         */
        void call_pycb();

        /*!
         * \brief python callback to be overwritten
         */
        virtual void call(std::vector<CompGraphCallbackValueProxy>&) = 0;
};

/*!
 * \brief wrap around shared mgb::ComputingGraph
 */
class CompGraph {
    class PyUserData;

    mgb::SmallVector<mgb::ComputingGraph::OutputSpec> m_out_specs;
    //! (callback, output spec part)
    mgb::SmallVector<std::pair<_CompGraphCallback*, size_t>> m_raw_callbacks;

    std::shared_ptr<mgb::ComputingGraph> m_comp_graph_own;
    std::weak_ptr<mgb::ComputingGraph> m_comp_graph_borrow;

    explicit CompGraph(const std::shared_ptr<mgb::ComputingGraph>& cg)
                : m_comp_graph_own{cg} {}

    explicit CompGraph(const std::weak_ptr<mgb::ComputingGraph> &cg):
        m_comp_graph_borrow{cg}
    {}

    public:

        CompGraph():
            m_comp_graph_own(mgb::ComputingGraph::make())
        {}

        // A mgb::cg::ComputingGraph may be wrapped in a CompGraph in two ways:
        // 1. Borrowing a ComputingGraph.
        // 2. Own a shared_ptr of ComputingGraph.
        // We make constructors private and use factory function instead to make
        // it explicit at the call site. (So-called "Named Constructor")

        /*!
         * \brief Wrap a ComputingGraph by borrowing a reference.
         */
        static CompGraph make_from_weak_ptr(
                const std::weak_ptr<mgb::ComputingGraph>& cg) {
            return CompGraph{cg};
        }

        /*!
         * \brief Wrap a ComputingGraph by owning one of its reference.
         */
        static CompGraph make_from_shared_ptr(
                const std::shared_ptr<mgb::ComputingGraph>& cg) {
            return CompGraph{cg};
        }

        CompGraph(const mgb::cg::SymbolVarArray& dest_symbol_vars) {
            m_comp_graph_own = mgb::ComputingGraph::make();
            mgb::cg::replace_vars_comp_graph(dest_symbol_vars,
                                                  m_comp_graph_own.get());
        }

        void clear_device_memory();

        //! get underlying ComputingGraph instance
        mgb::ComputingGraph& get() const;

        CompGraph& share_device_memory_with(CompGraph &other) {
            get().share_device_memory_with(other.get());
            return *this;
        }

        //! get a dict to store arbitrary user data
        PyObject* _user_data();

        AsyncExec _do_compile(bool copy, bool optimize_for_inference);
        std::vector<AsyncExec> _do_compile_multi_part();

        /*!
         * \brief add an output spec
         * \param callback callback to be invoked; or nullptr for computing
         *      output var only
         */
        void _add_output_spec(mgb::cg::SymbolVar &var,
                _CompGraphCallback *callback);

        //! mark currently added output specs as a part in multi-part compile
        void _add_multi_part_endpoint() {
            m_out_specs.emplace_back();
        }

        void _clear_output_spec() {
            m_raw_callbacks.clear();
            m_out_specs.resize(1);
            m_out_specs[0].clear();
        }

        size_t _release() {
            if (m_comp_graph_own) {
                auto ret = m_comp_graph_own.use_count();
                m_comp_graph_own.reset();
                return ret;
            }
            m_comp_graph_borrow.reset();
            return 0;
        }

};

//! wrap shared_ptr<DTypeScalar>
class SharedScalar {
    bool m_dtype_locked = false;
    std::shared_ptr<mgb::DTypeScalar> m_val;
    mgb::HostTensorND m_val_as_host_nd;
    mgb::CompNode::UnorderedMap<std::shared_ptr<mgb::DeviceTensorND>> m_dev_val;

    mgb::HostTensorND& val_as_host_nd();

    public:
        SharedScalar(PyObject *val);
        void _set(PyObject *val);
        PyObject* _get();
        mgb::SymbolVar _as_sym_var(CompGraph &cg, mgb::CompNode &cn);

        void _lock_dtype() {
            m_dtype_locked = true;
        }

        bool _dtype_locked() {
            return m_dtype_locked;
        }

        const std::shared_ptr<mgb::DTypeScalar>& get_val() const {
            return m_val;
        }
};

/*!
 * \brief wrap around shared mgb::cg::OperatorNodeBase
 */
class Operator {
    mgb::cg::OperatorNodeBase* m_operator_node;

public:
    Operator() : m_operator_node(nullptr){};
    Operator(mgb::cg::OperatorNodeBase* operator_node)
            : m_operator_node(operator_node) {}

    size_t id() const { return m_operator_node->id(); }

    const std::string& name() const { return m_operator_node->name(); }

    const std::shared_ptr<mgb::ComputingGraph> get_owner_graph() const {
        return m_operator_node->owner_graph()->shared_from_this();
    }

    const mgb::SymbolVarArray inputs() const {
        return mgb::cg::to_symbol_var_array(m_operator_node->input());
    }

    const mgb::SymbolVarArray outputs() const {
        return mgb::cg::to_symbol_var_array(m_operator_node->output());
    }

    mgb::cg::OperatorNodeBase* node() const { return m_operator_node; }
};

//! get inferred value as numpy ndarray or None
PyObject* get_symvar_inferred_value(mgb::SymbolVar var);

mgb::SymbolVar fill_retain_dtype(mgb::SymbolVar var, PyObject* value);

//! whether _mgb_global_finalize() has been called
bool global_finalized();

#ifndef SWIG
void mark_as_input(mgb::cg::ComputingGraph* cg, mgb::cg::SymbolVar var);
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
