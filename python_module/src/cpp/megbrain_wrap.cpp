/**
 * \file python_module/src/cpp/megbrain_wrap.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./megbrain_wrap.h"
#include "./python_helper.h"
#include "./megbrain_pubapi_internal.h"

#include "megbrain/version.h"
#include "megbrain/tensor.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/utils/thread.h"
#include "megbrain/utils/timer.h"

#include <cstring>
using namespace mgb;

namespace {
    bool g_global_finalize_called = false;

    /*!
     * \brief record the vars produced from user-created Host2DeviceCopy
     *
     * Note that the vars are mapped by address of underlying HostTensorND, so
     * in the case of partial execution, vars in the parent graph can be
     * retrieved from oprs in the sub graphs.
     */
    class UserInputVars final : public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;

        //! we keep this mapping to handle multi-part compiling, where new
        //! graphs would be created and the var in the original graph is needed
        ThinHashMap<HostTensorND*, VarNode*> m_tensor2var;

    public:
        void register_var(SymbolVar x) {
            m_tensor2var[x.node()->owner_opr()
                                  ->cast_final_safe<opr::Host2DeviceCopy>()
                                  .host_data()
                                  .get()] = x.node();
        }

        //! get the corresponding var from an opr if it has been registered;
        //! return nullptr otherwise
        VarNode* check(cg::OperatorNodeBase* opr) const {
            if (opr->same_type<opr::Host2DeviceCopy>()) {
                auto ptr = opr->cast_final<opr::Host2DeviceCopy>()
                                   .host_data()
                                   .get();
                auto iter = m_tensor2var.find(ptr);
                return iter == m_tensor2var.end() ? nullptr : iter->second;
            }
            return nullptr;
        }

        static UserInputVars& get(ComputingGraph* graph) {
            return *graph->options()
                            .user_data.get_user_data_or_create<UserInputVars>();
        }
    };

    __attribute__((constructor))
    void global_init() {
        CompNode::enable_affinity_for_cpu(true);
    }
} // anonymous namespace

MGB_TYPEINFO_OBJ_IMPL(UserInputVars);

/* ================= SharedND =================  */

bool SharedND::sync(mgb::DeviceTensorND &dv) {
    if (m_copy_sync) {
        dv.sync();
        return true;
    }
    return false;
}

void SharedND::_set_init_shape(const std::vector<size_t> &shape) {
    mgb_assert(m_dev_tensor && m_dev_tensor->empty());
    m_dev_tensor->resize(npy::vec2shape(shape));
}

void SharedND::_resize(const std::vector<size_t> &shape) {
    auto tshp = npy::vec2shape(shape);
    if (m_dev_tensor) {
        m_dev_tensor->resize(tshp);
    } else {
        mgb_assert(m_var);
        m_var->shape_alloc(tshp);
    }
}

void SharedND::_reset_zero() {
    fill_zero_dev_tensor(*m_dev_tensor);
}

void SharedND::_copy_from_npyarr(PyObject *npyarr) {
    auto do_copy = [&](DeviceTensorND *dest, VarNode *var) {
        DType dtype = dest ? dest->dtype() : var->dtype();
        mgb_assert(dtype.valid());
        auto hv = npy::np2tensor(npyarr, npy::Meth::borrow(), dtype);
        if (var) {
            // only setup by assign(), by craniotome
            var->shape_alloc(hv.shape());
            dest = &var->mutable_dev_tensor();
        }
        if (!sync(dest->copy_from(hv))) {
            m_async_copy_refkeeper = hv;
        } else {
            m_async_copy_refkeeper = {};
        }
    };
    if (m_var) {
        mgb_assert(!m_dev_tensor);
        do_copy(nullptr, m_var);
    } else {
        mgb_assert(m_dev_tensor);
        do_copy(m_dev_tensor.get(), nullptr);
    }
}

PyObject* SharedND::_get_npyarr() {
    mgb_assert(m_dev_tensor);
    if (m_dev_tensor->empty())
        Py_RETURN_NONE;
    HostTensorND hv;
    hv.comp_node(CompNode::default_cpu())
        .copy_from(*m_dev_tensor)
        .sync();
    return npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE);
}

PyObject* SharedND::_get_dtype() {
    mgb_assert(m_dev_tensor);
    return npy::dtype_mgb2np(m_dev_tensor->dtype());
}

void SharedND::_copy_from_value_proxy(CompGraphCallbackValueProxy &value) {
    if (value.eager_copy()) {
        mgb_log_warn("copy from eager-copied CompGraphCallbackValueProxy into"
                " SharedND; consider using callback_lazycopy; traceback:\n%s",
                PyStackExtracter::run().c_str());
    }

    if (m_var) {
        mgb_assert(!m_dev_tensor);
        auto &&src = value.dev_tensor();
        m_var->shape_alloc(src.shape()).
            mutable_dev_tensor().copy_from(src);
    } else {
        mgb_assert(m_dev_tensor);
        sync(m_dev_tensor->copy_from(value.dev_tensor()));
    }
}

void SharedND::_share_from_value_proxy(CompGraphCallbackValueProxy& value) {
    if (value.eager_copy()) {
        mgb_log_warn(
                "share value from eager-copied CompGraphCallbackValueProxy into"
                " SharedND; consider using callback_lazycopy; traceback:\n%s",
                PyStackExtracter::run().c_str());
    }

    if (m_var) {
        mgb_assert(!m_dev_tensor);
        m_var->reset_dev_tensor_from_tensor(value.dev_tensor());
    } else {
        mgb_assert(m_dev_tensor);
        *m_dev_tensor = value.dev_tensor();
    }
}

SharedND SharedND::_from_symvar(SymbolVar symvar) {
    auto opr = symvar.node()->owner_opr();
    if (auto vsnd = opr->try_cast_final<opr::VolatileSharedDeviceTensor>()) {
        return SharedND(vsnd->dev_data());
    }
    if (auto snd = opr->try_cast_final<opr::SharedDeviceTensor>()) {
        return SharedND(snd->dev_data());
    }
    mgb_throw(MegBrainError, "cannot convert from %s", opr->dyn_typeinfo()->name);
}

uintptr_t SharedND::_pubapi_dev_tensor_ptr(int version) {
    DeviceTensorND *dv;
    if (m_dev_tensor) {
        mgb_assert(!m_var);
        dv = m_dev_tensor.get();
    } else {
        mgb_assert(m_var);
        dv = nullptr;
    }
    void *ret;
    if (version == 0) {
        if (dv) {
            ret = dv->raw_ptr();
        } else {
            ret = m_var->dev_tensor().raw_ptr();
        }
    } else {
        init_pubapi_dev_tensor(m_pubapi_dev_tensor, dv, m_var, false);
        ret = &m_pubapi_dev_tensor;
    }
    return reinterpret_cast<uintptr_t>(ret);
}

SymbolVar SharedND::_as_sym_var(CompGraph &cg, const std::string &name,
        bool volatile_) {
    mgb_assert(m_dev_tensor);
    OperatorNodeConfig config;
    if (!name.empty())
        config.name(name);
    if (volatile_) {
        return opr::VolatileSharedDeviceTensor::make(cg.get(), m_dev_tensor,
                config);
    } else {
        return opr::SharedDeviceTensor::make(cg.get(), m_dev_tensor, config);
    }
}

std::vector<size_t> SharedND::_get_shape(){
    if (m_var) {
        mgb_assert(!m_dev_tensor);
        return npy::shape2vec(m_var->shape());
    }
    mgb_assert(m_dev_tensor);
    return npy::shape2vec(m_dev_tensor->shape());
}

void SharedND::copy_to_sub_from_shared(
        int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step,
        const SharedND &rhs) {
    mgb_assert(m_dev_tensor && rhs.m_dev_tensor);
    auto sub = m_dev_tensor->sub(
            Slice(begin, end, step).apply(m_dev_tensor->layout(), axis));
    sub.copy_from_fixlayout(*rhs.m_dev_tensor).sync();

}

void SharedND::copy_from_shared_sub(const SharedND &rhs,
        int axis, ptrdiff_t begin, ptrdiff_t end, ptrdiff_t step) {
    mgb_assert(m_dev_tensor && rhs.m_dev_tensor);
    if (axis == -3) {
        sync(m_dev_tensor->copy_from_fixlayout(*rhs.m_dev_tensor));
    } else if (axis == -2) {
        sync(m_dev_tensor->copy_from(*rhs.m_dev_tensor));
    } else {
        auto sub = rhs.m_dev_tensor->sub(
                Slice(begin, end, step).apply(
                    rhs.m_dev_tensor->layout(), axis));
        sync(m_dev_tensor->copy_from(sub));
    }
}

void SharedND::_check_before_share_memory(const SharedND& rhs) {
    mgb_assert(rhs.m_dev_tensor);
    mgb_assert(m_dev_tensor);
    mgb_assert(rhs.m_dev_tensor->dtype() == m_dev_tensor->dtype());
    mgb_assert(rhs.m_dev_tensor->comp_node() == m_dev_tensor->comp_node());
}

void SharedND::_share_memory_from(const SharedND& rhs, size_t begin) {
    _check_before_share_memory(rhs);
    m_dev_tensor->reset(
        rhs.m_dev_tensor->storage().sub(m_dev_tensor->dtype().size() * begin),
        m_dev_tensor->layout());
}

void SharedND::_reset_dev_tensor(const SharedND &rhs) {
    _check_before_share_memory(rhs);
    *m_dev_tensor = *(rhs.m_dev_tensor);
}

/* ================= _HostSharedND =================  */

void _HostSharedND::ensure_own_storage() {
    if (!m_own_storage) {
        mgb_assert(m_tensor);
        HostTensorND val{m_tensor->comp_node(), m_tensor->dtype()};
        if (!m_tensor->empty()) {
            val.resize(m_tensor->shape());
        }
        *m_tensor = std::move(val);
        m_own_storage = true;
    }
}

void _HostSharedND::_resize(const std::vector<size_t> &shape) {
    ensure_own_storage();
    m_tensor->resize(npy::vec2shape(shape));
}

void _HostSharedND::_copy_from_npyarr(PyObject *npyarr, bool borrow) {
    mgb_assert(m_tensor);
    mgb_assert(m_tensor->dtype().valid());
    if (!m_borrow_on_cpu &&
            m_tensor->comp_node().device_type() == CompNode::DeviceType::CPU) {
        borrow = false;
    }
    if (borrow) {
        auto val = npy::np2tensor(
                npyarr, npy::Meth::borrow(m_tensor->comp_node()),
                m_tensor->dtype());
        m_own_storage = false;
        *m_tensor = std::move(val);
    } else {
        ensure_own_storage();
        npy::np2tensor(npyarr,
                npy::Meth::copy_into(m_tensor.get()), m_tensor->dtype());
    }
}

SymbolVar _HostSharedND::_as_sym_var(CompGraph &cg, bool enable_static_infer,
        const std::string &name) {
    if (m_tensor->empty())
        cg.get().options().allocate_static_mem_after_graph_compile = false;

    OperatorNodeConfig config;
    if (!name.empty())
        config.name(name);

    SymbolVar ret;
    if (enable_static_infer) {
        ret = opr::Host2DeviceCopy::make(cg.get(), m_tensor, config);
    } else {
        ret = opr::Host2DeviceCopy::make_no_value_infer(cg.get(), m_tensor,
                config);
    }
    UserInputVars::get(&cg.get()).register_var(ret);
    return ret;
}

_HostSharedND _HostSharedND::make_proxy(SymbolVar var) {
    auto &&opr = var.node()->owner_opr()->
       cast_final_safe<opr::Host2DeviceCopy>();
    _HostSharedND rst{var.node()->comp_node(), var.dtype()};
    rst.m_tensor = opr.host_data();
    rst.m_proxied_opr = &opr;
    return rst;
}

std::string _HostSharedND::__repr__() const {
    if (m_proxied_opr) {
        return ssprintf("<HostSharedND proxy at %p for %s>",
                this, m_proxied_opr->cname());
    }
    return ssprintf("<HostSharedND at %p>", this);
}

PyObject* _HostSharedND::_get_dtype() {
    mgb_assert(m_tensor);
    return npy::dtype_mgb2np(m_tensor->dtype());
}

/* ================= CompGraphCallbackValueProxy =================  */

CompGraphCallbackValueProxy
CompGraphCallbackValueProxy::make_raw_host_value_proxy(
        const mgb::HostTensorND &hv) {
    CompGraphCallbackValueProxy ret;
    ret.m_use_raw_hv = true;
    ret.m_hv = hv;
    ret.m_is_active = true;
    return ret;
}

void CompGraphCallbackValueProxy::setup(
        const mgb::DeviceTensorND &val, bool eager_copy) {

    while (__atomic_load_n(&m_is_active, __ATOMIC_SEQ_CST)) {
        // wait for previous callback to finish
        std::this_thread::yield();
    }

    mgb_assert(!m_use_raw_hv && val.shape_valid());
    m_eager_copy = eager_copy;
    m_dev_value = val;
    if (eager_copy) {
        m_value_used = false;
        do_copy();
    } else {
        m_value_used = true;
    }

    __atomic_store_n(&m_is_active, true, __ATOMIC_SEQ_CST);
}

void CompGraphCallbackValueProxy::do_copy() {
    mgb_assert(!m_use_raw_hv && m_dev_value.shape_valid());
    m_hv.copy_from(m_dev_value);
    auto cn = m_hv.comp_node();
    if (!m_copy_event)
        m_copy_event = cn.create_event();
    m_copy_event->record();
}

void CompGraphCallbackValueProxy::sync() {
    mgb_assert(!m_use_raw_hv);
    RealTimer t0;
    double next_warn_time = 2, warn_time_delta = 1;
    while (!m_copy_event->finished()) {
        usleep(1);
        if (t0.get_secs() >= next_warn_time) {
            mgb_log_warn("wait d2h copy for more than %.3f secs",
                    t0.get_secs());
            next_warn_time += warn_time_delta;
            warn_time_delta += 1;
        }
    }
}

void CompGraphCallbackValueProxy::on_finished() {
    mgb_assert(m_is_active && !m_use_raw_hv);
    m_dev_value = {};
    if (m_hv.shape_valid()) {
        m_hv.resize({});    // resize to reuse buffer
    }
    __atomic_store_n(&m_is_active, false, __ATOMIC_SEQ_CST);
    if (!m_value_used) {
        mgb_log_warn("computing graph callback did not read the value");
    }
}

PyObject* CompGraphCallbackValueProxy::_get_npyarr() {
    mgb_assert(m_is_active);

    if (!m_use_raw_hv) {
        mgb_assert(m_dev_value.shape_valid());
        if (!m_hv.shape_valid()) {
            do_copy();
            sync();
        }
    }
    m_value_used = true;
    return npy::ndarray_from_tensor(m_hv, npy::ShareType::TRY_SHARE);
}

PyObject* CompGraphCallbackValueProxy::_get_dtype() {
    mgb_assert(m_is_active);

    if (m_use_raw_hv)
        return npy::dtype_mgb2np(m_hv.dtype());

    mgb_assert(m_dev_value.shape_valid());
    return npy::dtype_mgb2np(m_dev_value.dtype());
}

std::vector<size_t> CompGraphCallbackValueProxy::_get_shape() {
    mgb_assert(m_is_active);

    if (m_use_raw_hv)
        return npy::shape2vec(m_hv.shape());

    mgb_assert(m_dev_value.shape_valid());
    return npy::shape2vec(m_dev_value.shape());
}

uintptr_t CompGraphCallbackValueProxy::_pubapi_dev_tensor_ptr(int version) {
    mgb_assert(m_is_active && !m_use_raw_hv);
    mgb_assert(m_dev_value.shape_valid());
    void *ret;
    if (version == 0) {
        ret = m_dev_value.raw_ptr();
    } else {
        init_pubapi_dev_tensor(
                m_pubapi_dev_tensor, &m_dev_value, nullptr, true);
        ret = &m_pubapi_dev_tensor;
    }
    return reinterpret_cast<uintptr_t>(ret);
}

mgb::CompNode CompGraphCallbackValueProxy::_get_comp_node() {
    mgb_assert(m_is_active && !m_use_raw_hv);
    mgb_assert(m_dev_value.shape_valid());
    return m_dev_value.comp_node();
}

/* ================= AsyncExec =================  */

class AsyncExec::Core {
    public:
        Core(std::unique_ptr<mgb::cg::AsyncExecutable> f):
            m_func(std::move(f))
        {
        }

        mgb::cg::AsyncExecutable* func() const {
            return m_func.get();
        }

        struct CallbackParam {
            std::vector<CompGraphCallbackValueProxy> value;
            _CompGraphCallback *cb;
        };

        void dispatch_callback(const CallbackParam &param) {
            m_worker.add_task(param);
        }

        void wait_callback_finish() {
            m_worker.wait_all_task_finish();
        }

    private:
        std::unique_ptr<mgb::cg::AsyncExecutable> m_func;

        class Worker final: public AsyncQueueSC<CallbackParam, Worker> {
            public:
                void process_one_task(CallbackParam &task) {
                    for (auto &tmp_value: task.value) {
                        tmp_value.sync();
                    }
                    task.cb->call_pycb();
                }
        };
        Worker m_worker;
};

AsyncExec::AsyncExec(std::unique_ptr<mgb::cg::AsyncExecutable> f):
    m_core(std::make_shared<Core>(std::move(f)))
{
}

AsyncExec::~AsyncExec() {
    if (m_core)
        _wait();
}

AsyncExec::Core* AsyncExec::core() const {
    return m_core.get();
}

void AsyncExec::_execute() {
    m_core->func()->execute();
}

std::string AsyncExec::_to_json_str() {
    auto jv = m_core->func()->to_json();
    return jv->to_string();
}

void AsyncExec::_wait() {
    m_core->wait_callback_finish();
    m_core->func()->wait();
}

double AsyncExec::_get_prev_exec_time() {
    return m_core->func()->get_prev_exec_time();
}

SymbolVarArray AsyncExec::_find_mutable_input() {
    ThinHashSet<VarNode*> used_set;
    UserInputVars* user_vars = nullptr;
    auto cb = [&](cg::OperatorNodeBase* opr) {
        if (!user_vars) {
            ComputingGraph* g;
            if (m_multi_part_par_graph)
                g = m_multi_part_par_graph.get();
            else
                g = opr->owner_graph();
            user_vars = &UserInputVars::get(g);
        }
        if (auto var = user_vars->check(opr)) {
            used_set.insert(var);
        }
        return true;
    };
    m_core->func()->iter_opr_seq(cb);
    for (auto i : m_core->func()->get_rt_static_source_deps()) {
        cb(i.dest->owner_opr());
    }
    SymbolVarArray ret;
    ret.reserve(used_set.size());
    ret.insert(ret.begin(), used_set.begin(), used_set.end());
    return ret;
}

void AsyncExec::clear_device_memory() {
    _wait();
    m_core->func()->clear_device_memory();
}

std::vector<std::pair<CompNode, size_t>>
AsyncExec::_update_static_alloc_plan_and_get_size() {
    std::vector<std::pair<CompNode, size_t>> ret;
    for (auto&& i : m_core->func()->update_static_alloc_plan_and_get_size()) {
        ret.emplace_back(i.first, i.second);
    }
    return ret;
}

/* ================= _CompGraphCallback =================  */

void _CompGraphCallback::set_async_exec(const AsyncExec &ae)  {
    mgb_assert(!m_ae_core);
    m_ae_core = ae.core();
}

void _CompGraphCallback::set_eager_copy(bool flag) {
    mgb_assert(!m_cb_created);
    m_eager_copy = flag;
}

std::function<void(mgb::SmallVector<mgb::DeviceTensorND> &)> _CompGraphCallback::make_multi_input_callback() {
    mgb_assert(!m_cb_created);
    m_cb_created = true;

    // shared_ptr would delete this afterwards
    std::shared_ptr <_CompGraphCallback> self(this);

    auto cb = [self](SmallVector <mgb::DeviceTensorND> &data) {
        for (size_t i = self->m_value_proxies.size(); i < data.size(); ++i) {
            self->m_value_proxies.emplace_back();
        }
        if (self->m_eager_copy) {
            mgb_assert(self->m_ae_core);
            for (size_t i = 0; i < self->m_value_proxies.size(); ++i) {
                self->m_value_proxies[i].setup(data[i], true);
            }
            self->m_ae_core->dispatch_callback(
                    AsyncExec::Core::CallbackParam{self->m_value_proxies, self.get()}
            );
        } else {
            for (size_t i = 0; i < self->m_value_proxies.size(); ++i)
                self->m_value_proxies[i].setup(data[i], false);
            self->call_pycb();
        }
    };

    return cb;
}

std::function<void(mgb::DeviceTensorND &)> _CompGraphCallback::make_callback() {
    this->m_value_proxies.emplace_back();
    mgb_assert(!m_cb_created);
    m_cb_created = true;

    // shared_ptr would delete this afterwards
    std::shared_ptr <_CompGraphCallback> self(this);

    auto cb = [self](mgb::DeviceTensorND &data) {
        if (self->m_eager_copy) {
            mgb_assert(self->m_ae_core);
            self->m_value_proxies[0].setup(data, true);
            self->m_ae_core->dispatch_callback(
                    AsyncExec::Core::CallbackParam{self->m_value_proxies, self.get()}
            );
        } else {
            self->m_value_proxies[0].setup(data, false);
            self->call_pycb();
        }
    };

    return cb;
}

void _CompGraphCallback::call_pycb() {
    try {
        call(m_value_proxies);
    } catch (...) {
        for(auto &m_value_proxy: m_value_proxies) {
            m_value_proxy.on_finished();
        }
        throw;
    }
    for(auto &m_value_proxy: m_value_proxies) {
        m_value_proxy.on_finished();
    }
}

/* ================= CompGraph =================  */

class CompGraph::PyUserData final: public UserDataContainer::UserData,
                                   public NonCopyableObj {
    MGB_TYPEINFO_OBJ_DECL;

    PyObject *m_obj;

    public:

        PyUserData() {
            PYTHON_GIL;
            m_obj = PyDict_New();
            mgb_assert(m_obj, "failed to create python object");
        }

        ~PyUserData() {
            PYTHON_GIL;
            Py_DECREF(m_obj);
        }

        PyObject* get() const {
            return m_obj;
        }
};
MGB_TYPEINFO_OBJ_IMPL(CompGraph::PyUserData);

mgb::ComputingGraph& CompGraph::get() const {
    if (m_comp_graph_own)
        return *m_comp_graph_own;
    auto &&val = m_comp_graph_borrow.lock();
    mgb_assert(val, "CompGraph has been destructed");
    return *val;
}

void CompGraph::clear_device_memory() {
    if (!m_comp_graph_own)
        return;
    m_comp_graph_own->clear_device_memory();
}

PyObject* CompGraph::_user_data() {
    auto ct = get().options().user_data.get_user_data_or_create<PyUserData>();
    auto ret = ct->get();
    PYTHON_GIL;
    Py_INCREF(ret);
    return ret;
}

void CompGraph::_add_output_spec(
        mgb::cg::SymbolVar &var, _CompGraphCallback *callback) {

    cg::ComputingGraph::Callback cb;
    if (callback) {
        cb = callback->make_callback();
        m_raw_callbacks.push_back({callback, m_out_specs.size() - 1});
    }
    if (m_out_specs.empty()) {
        m_out_specs.emplace_back();
    }
    m_out_specs.back().push_back({var, cb});
}

AsyncExec CompGraph::_do_compile(bool copy, bool optimize_for_inference) {
    mgb_assert(m_out_specs.size() == 1, "got %zu output specs for compile",
               m_out_specs.size());
    auto&& spec = m_out_specs[0];
    if (optimize_for_inference) {
        SymbolVarArray vars;
        vars.reserve(spec.size());
        for (auto&& i : spec) {
            vars.push_back(i.first);
        }
        vars = gopt::optimize_for_inference(vars, {});
        mgb_assert(vars.size() == spec.size());
        for (size_t i = 0; i < vars.size(); ++i) {
            spec[i].first = vars[i];
        }
    }

    std::unique_ptr<mgb::cg::AsyncExecutable> async_executable;
    if (get().options().eager_evaluation ||
        (copy && get().current_comp_seq())) {
        // need to copy a new comp graph
        SymbolVarArray vars;
        vars.reserve(spec.size());
        for (auto&& i : spec) {
            vars.emplace_back(i.first);
        }

        // copy graph
        auto new_graph = mgb::ComputingGraph::make();
        SymbolVarArray new_vars =
                replace_vars_comp_graph(std::move(vars), new_graph.get());
        mgb_assert(new_vars.size() == spec.size());

        // register input
        auto h2d = find_h2d(new_vars);
        for (auto&& i : h2d) {
            UserInputVars::get(new_graph.get()).register_var(i);
        }

        mgb::ComputingGraph::OutputSpec new_spec;
        new_spec.reserve(spec.size());
        for (size_t i = 0; i < spec.size(); ++i) {
            new_spec.emplace_back(mgb::ComputingGraph::OutputSpecItem{
                    new_vars[i], spec[i].second});
        }
        async_executable = new_graph->compile(new_spec);
    } else {
        async_executable = get().compile(spec);
    }

    AsyncExec ret{std::move(async_executable)};

    for (auto&& i : m_raw_callbacks) {
        mgb_assert(!i.second);
        i.first->set_async_exec(ret);
    }
    _clear_output_spec();
    return ret;
}

std::vector<AsyncExec> CompGraph::_do_compile_multi_part() {
    // last spec is empty due to an extra call to _add_multi_part_endpoint()
    mgb_assert(m_out_specs.size() > 1 && m_out_specs.back().empty(),
               "got %zu output specs for multi-part compile",
               m_out_specs.size());
    m_out_specs.pop_back();
    std::vector<AsyncExec> ret;
    ret.reserve(m_out_specs.size());
    auto graph = get().shared_from_this();
    for (auto&& i : graph->compile_multi_part(m_out_specs)) {
        ret.emplace_back(std::move(i));
    }
    for (auto&& i : ret) {
        i.set_multi_part_par_graph(graph);
    }
    for (auto&& i : m_raw_callbacks) {
        i.first->set_async_exec(ret.at(i.second));
    }
    _clear_output_spec();
    return ret;
}

/* ================= SharedScalar =================  */

SharedScalar::SharedScalar(PyObject *val):
    m_val{std::make_shared<DTypeScalar>()}
{
    _set(val);
}

HostTensorND& SharedScalar::val_as_host_nd() {
    if (m_val_as_host_nd.empty()) {
        HostTensorStorage storage;
        storage.reset(CompNode::default_cpu(), m_val->dtype().size(),
                      {m_val, static_cast<dt_byte*>(
                                      const_cast<void*>(m_val->storage()))});
        m_val_as_host_nd.reset(storage, {TensorShape{1}, m_val->dtype()});
    }
    return m_val_as_host_nd;
}

void SharedScalar::_set(PyObject *val) {
    auto tensor = npy::np2tensor(val, npy::Meth::borrow(), {});
    mgb_assert(tensor.layout().is_scalar(),
            "value given to SharedScalar must be scalar; got shape %s",
            tensor.shape().to_string().c_str());
    if (m_dtype_locked) {
        mgb_assert(tensor.dtype() == m_val->dtype(),
                "dtype for SharedScalar has been locked as %s, "
                "but attempt to set it to %s", m_val->dtype().name(),
                tensor.dtype().name());
    }
    m_val->set_raw(tensor.dtype(), tensor.raw_ptr());

    if (!m_dev_val.empty()) {
        auto &&hv = val_as_host_nd();
        for (auto &&i: m_dev_val)
            i.second->copy_from_fixlayout(hv);
    }
}

PyObject* SharedScalar::_get() {
    HostTensorND hv{CompNode::default_cpu(), TensorShape{1}, m_val->dtype()};
    memcpy(hv.raw_ptr(), m_val->storage(), m_val->dtype().size(1));
    return npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE);
}

SymbolVar SharedScalar::_as_sym_var(CompGraph &cg, mgb::CompNode &cn) {
    m_dtype_locked = true;
    auto &&dv = m_dev_val[cn];
    auto &&hv = val_as_host_nd();
    if (!dv) {
        dv = std::make_shared<DeviceTensorND>(cn);
        dv->copy_from(hv);
    }
    return opr::SharedDeviceTensor::make(cg.get(), dv,
            ssprintf("SharedScalar@%p", m_val.get()));
}

/* ================= misc =================  */

SymbolVar fill_retain_dtype(SymbolVar var, PyObject *value) {
    auto tensor = npy::np2tensor(value, npy::Meth::borrow(), {});
    mgb_assert(tensor.shape().is_scalar(),
            "value for fill_retain_dtype must be scalar; got shape %s",
            tensor.shape().to_string().c_str());
    switch (tensor.dtype().enumv()) {
#define cb(_dt) case DTypeTrait<_dt>::enumv: \
        static_assert(sizeof(DTypeTrait<_dt>::ctype) <= sizeof(int), \
                "bad dtype size"); \
        return var.fill_retain_dtype(static_cast<int>( \
                    *tensor.ptr<DTypeTrait<_dt>::ctype>()));
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
        case DTypeEnum::Float32:
            return var.fill_retain_dtype(*tensor.ptr<dt_float32>());
        case DTypeEnum::Float16:
            return var.fill_retain_dtype(
                    static_cast<float>(*tensor.ptr<dt_float16>()));
        // TODO: What does this mean?
        case DTypeEnum::Quantized8Asymm:
        case DTypeEnum::QuantizedS32:
        case DTypeEnum::QuantizedS8:
        case DTypeEnum::Quantized4Asymm:
        case DTypeEnum::QuantizedS4:
        case DTypeEnum::Byte:
        case DTypeEnum::QuantizedS16:
            break;
#define cb(low_bit, size) \
        case DTypeEnum::low_bit##size: \
            break;
MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb

    }
    throw ConversionError(ssprintf(
                "unsupported value dtype: %s", tensor.dtype().name()));
}

PyObject* get_symvar_inferred_value(mgb::SymbolVar symvar) {
    auto var = symvar.node();
    auto&& mgr = var->owner_graph()->static_infer_manager();
    using IT = cg::static_infer::InferType;
    auto it = mgr.get_infer_type(var);
    if (!(it.value & (IT::CONST | IT::RT_STATIC)))
        Py_RETURN_NONE;

    auto val = mgr.infer_value_fallible(var);
    if (!val)
        Py_RETURN_NONE;

    auto hv = HostTensorND::make_proxy(*val);
    return npy::ndarray_from_tensor(hv, npy::ShareType::MUST_UNSHARE);
}

void _mgb_global_finalize() {
    CompNode::finalize();
    g_global_finalize_called = true;
}

bool global_finalized() {
    return g_global_finalize_called;
}

std::vector<size_t> _get_mgb_version() {
    return {MGB_MAJOR, MGB_MINOR, MGB_PATCH, MGB_IS_DEV};
}

SymbolVarArray _grad(SymbolVar target, SymbolVarArray wrts,
        bool warn_mid_wrt, int use_virtual_grad,
        bool return_zero_for_nodep) {
    if (use_virtual_grad == -1) {
        use_virtual_grad = std::abs(
                target.node()->owner_graph()->options().graph_opt_level) >= 2;
    }

    if (use_virtual_grad) {
        mgb_assert(return_zero_for_nodep,
            "can't return a null var when using virtual grad opr");
        SymbolVarArray ret;
        ret.reserve(wrts.size());
        for (auto&& wrt : wrts) {
            ret.push_back(opr::VirtualGrad::make(target, wrt));
        }
        return ret;
    }
    return cg::grad(target, wrts, warn_mid_wrt, return_zero_for_nodep);
}

SymbolVar _inter_graph_trans_var(
        CompGraph &dest_graph, SymbolVar src) {
    auto &&graph = dest_graph.get();
    auto trans = mgb::cg::InterGraphVarTransformer::get(graph);
    mgb_assert(trans, "trans func on graph %p has not been setup", &graph);
    return trans->trans(src.node());
}

SymbolVar _get_graph_optimizer_replaced_var(SymbolVar src) {
    return gopt::GraphOptimizer::var_replace_lookup(src.node());
}

void mark_as_input(ComputingGraph* cg, SymbolVar var) {
    VarNode* node = var.node();
    mgb_assert(node->owner_graph() == cg);
    mgb_assert(node->owner_opr()->same_type<opr::Host2DeviceCopy>());
    UserInputVars::get(cg).register_var(var);
}

namespace {

void add_update_impl(const DeviceTensorND& dest,
        const DeviceTensorND& delta_nobrd,
        float alpha, float beta, float bias) {
    auto&& cn = dest.comp_node();
    using DT = CompNode::DeviceType;
    mgb_assert(cn == delta_nobrd.comp_node() &&
        (cn.device_type() == DT::CUDA || cn.device_type() == DT::CPU));
    mgb_assert(dest.dtype() == delta_nobrd.dtype());
    auto&& delta = delta_nobrd.sub(SubTensorSpec::make_from_offset_elem(
        delta_nobrd.layout().broadcast(dest.shape()), 0));
    cn.activate();
    if (!static_cast<bool>(alpha) && beta == 1 &&
            !static_cast<bool>(bias)) {
        dest.copy_from_fixlayout(delta);
    } else {
        auto&& handle = MegDNNHandle::get(
                CompNodeEnv::from_comp_node(cn)).handle();
        auto&& op = handle->create_operator<megdnn::AddUpdate>();
        op->param() = {alpha, beta, bias};
        op->exec(dest.as_megdnn(), delta.as_megdnn());
        if (cn.device_type() == DT::CPU && cn != CompNode::default_cpu()) {
            CompNodeEnv::from_comp_node(cn).cpu_env().dispatch(
                [p = op.release()] { delete p; }
            );
        }
    }
}

} // anonymous namespace

void _add_update_fastpath(SharedND& dest_, SharedND& delta_,
        float alpha, float beta, float bias) {
    auto&& dest = dest_.dev_tensor();
    auto&& delta = delta_.dev_tensor();
    add_update_impl(*dest, *delta, alpha, beta, bias);
}

void _add_update_fastpath(SharedND& dest_, CompGraphCallbackValueProxy& delta_,
        float alpha, float beta, float bias) {
    auto&& dest = dest_.dev_tensor();
    auto&& delta = delta_.dev_tensor();
    add_update_impl(*dest, delta, alpha, beta, bias);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
