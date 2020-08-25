/**
 * \file python_module/src/cpp/plugin.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief helpers for debugging
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./plugin.h"
#include "./python_helper.h"

#include "megbrain/system.h"

#include <thread>
#include <cstring>
#include <sstream>

#ifdef WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif
#include <signal.h>

/* ================= _InfkernFinderImpl ================= */
size_t _InfkernFinderImpl::sm_id = 0;

_InfkernFinderImpl::_InfkernFinderImpl(CompGraph &cg, bool record_input_value):
    m_id{sm_id ++},
    m_comp_graph{cg.get().shared_from_this()},
    m_finder{m_comp_graph.get(), record_input_value}
{
}

size_t _InfkernFinderImpl::_write_to_file(const char *fpath) {
    auto opr = m_finder.write_to_file(fpath);
    if (opr)
        return opr->id() + 1;
    return 0;
}

size_t _InfkernFinderImpl::_get_input_values_prepare(size_t opr_id) {
    m_inp_val = m_finder.get_input_values(opr_id);
    return m_inp_val.size();
}

const char* _InfkernFinderImpl::_get_input_values_var_name(size_t idx) {
    return m_inp_val.at(idx).first->cname();
}

size_t _InfkernFinderImpl::_get_input_values_var_idx(size_t idx) {
    return m_inp_val.at(idx).first->id();
}

size_t _InfkernFinderImpl::_get_input_values_run_id(size_t idx) {
    return m_inp_val.at(idx).second.run_id;
}

CompGraphCallbackValueProxy  _InfkernFinderImpl::_get_input_values_val(size_t idx) {
    return CompGraphCallbackValueProxy::make_raw_host_value_proxy(
            m_inp_val.at(idx).second.val);
}

std::string _InfkernFinderImpl::__repr__() {
    return mgb::ssprintf(
            "_InfkernFinderImpl(%zu,graph=%p)", m_id, m_comp_graph.get());
}

/* ================= _FastSignal ================= */

class _FastSignal::Impl {
    using HandlerCallback = std::function<void()>;
    bool m_worker_started = false;
    std::mutex m_mtx;
    std::thread m_worker_hdl;
#ifdef WIN32
    SECURITY_ATTRIBUTES win_sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
    HANDLE pipe_r, pipe_w;
    DWORD bytes_r_w;
#else
    int m_pfd[2]; //! pipe fds; write signal handlers, -1 for exit
#endif
    std::unordered_map<int, HandlerCallback> m_handler_callbacks;

    void worker() {
        std::ostringstream oss;
        oss << std::this_thread::get_id() << std::endl;
        mgb_log("fast signal worker started in thread %s", oss.str().c_str());
        mgb::sys::set_thread_name("fastsgl");
        int signum;
        for (;;) {
#ifdef WIN32
            if (ReadFile(pipe_r, &signum, sizeof(int), &bytes_r_w, NULL) ==
                NULL) {
#else
            if (read(m_pfd[0], &signum, sizeof(int)) != sizeof(int)) {
#endif
                if (errno == EINTR)
                    continue;
                mgb_log_error("fast signal worker: "
                        "failed to read from self pipe: %s",
                        strerror(errno));
                return;
            }
            std::exception_ptr exc_ptr;
            if (signum == -1)
                return;
            try {
                HandlerCallback *cb;
                {
                    MGB_LOCK_GUARD(m_mtx);
                    cb = &m_handler_callbacks.at(signum);
                }
                (*cb)();
            } MGB_CATCH_ALL_EXCEPTION("fast signal worker", exc_ptr);
        }
    }

    void setup() {
        if (m_worker_started)
            return;

#ifdef WIN32
        if (!CreatePipe(&pipe_r, &pipe_w, &win_sa, 0)) {
            throw mgb::MegBrainError(mgb::ssprintf("failed to create pipe: %s",
                                                   strerror(errno)));
        }
#else
        if (pipe(m_pfd)) {
            throw mgb::MegBrainError(mgb::ssprintf("failed to create pipe: %s",
                                                   strerror(errno)));
        }
#endif
        std::thread t(std::bind(&Impl::worker, this));
        m_worker_hdl.swap(t);
        m_worker_started = true;
    }

    void write_pipe(int v) {
        mgb_assert(m_worker_started);
#ifdef WIN32
        if (WriteFile(pipe_w, &v, sizeof(int), &bytes_r_w, NULL) == NULL) {
#else
        if (write(m_pfd[1], &v, sizeof(int)) != sizeof(int)) {
#endif
            mgb_log_error("fast signal: failed to write to self pipe: %s",
                    strerror(errno));
        }
    }

    public:
        bool worker_started() const {
            return m_worker_started;
        }

        void register_handler(int signum, PyObject *func) {
            setup();

            {
                PYTHON_GIL;
                mgb_assert(PyCallable_Check(func));
                Py_INCREF(func);
            }
            auto deleter = [](PyObject *f){
                PYTHON_GIL;
                Py_DECREF(f);
            };
            std::shared_ptr<PyObject> funcptr(func, deleter);

            auto callback = [funcptr]() {
                PYTHON_GIL;
                auto func = funcptr.get();
                auto ret = PyObject_CallObject(func, nullptr);
                mgb_assert(ret, "failed to call pyobj %p; repr=%s",
                        func, PyUnicode_AsUTF8(PyObject_Repr(func)));
                Py_DECREF(ret);
            };

            MGB_LOCK_GUARD(m_mtx);
            m_handler_callbacks[signum] = callback;
        }

        void shutdown() {
            MGB_LOCK_GUARD(m_mtx);
            if (!m_worker_started)
                return;
            write_pipe(-1);
            m_worker_hdl.join();
#ifdef WIN32
            CloseHandle(pipe_r);
            CloseHandle(pipe_w);
#else
            close(m_pfd[0]);
            close(m_pfd[1]);
#endif
            m_handler_callbacks.clear();
            m_worker_started = false;
        }

        void signal_hander(int signum) {
            write_pipe(signum);
        }

        ~Impl() {
            shutdown();
        }
};

_FastSignal::Impl _FastSignal::sm_impl;

void _FastSignal::signal_hander(int signum) {
    if (sm_impl.worker_started())
        sm_impl.signal_hander(signum);
}

void _FastSignal::register_handler(int signum, PyObject *func) {
#ifdef WIN32
    //! up to now we can only use CTRL_C_EVENT to unix signal.SIGUSR1/2
    //FIXME: how to coherence signal number at python side
    // https://docs.microsoft.com/en-gb/cpp/c-runtime-library/reference/signal?view=vs-2017
    mgb_assert(signum == CTRL_C_EVENT, "only allow register CTRL_C_EVENT as unix signal.SIGUSR1/2 now");
    signal(signum, signal_hander);
#else
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = &signal_hander;
    int ret = sigaction(signum, &action, nullptr);
    mgb_assert(!ret, "sigaction failed: %s", strerror(errno));
#endif

    sm_impl.register_handler(signum, func);
}

void _FastSignal::shutdown()  {
    sm_impl.shutdown();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

