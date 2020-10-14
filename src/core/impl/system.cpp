/**
 * \file src/core/impl/system.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/system.h"
#include "megbrain/common.h"
#include "megbrain/utils/thin/hash_table.h"

#include <thread>

using namespace mgb;
using namespace sys;

int sys::get_cpu_count() {
    return std::max(std::thread::hardware_concurrency(), 1u);
}

#if defined(WIN32)

#include <windows.h>
void sys::set_cpu_affinity(const std::vector<int> &cpuset) {
    mgb_log_warn("Set_cpu_affinity will not support later");
    auto nr = get_cpu_count();
    DWORD mask = 0;
    for (auto i: cpuset) {
        mgb_assert(i >= 0 && i < 64 && i < nr);
        mask |= 1 << i;
    }
    auto succ = SetThreadAffinityMask(GetCurrentThread(), mask);
    if (!succ) {
        mgb_log_error("SetThreadAffinityMask failed (error ignored)");
    }
}

std::pair<size_t, size_t> sys::get_ram_status_bytes() {
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    auto succ = GlobalMemoryStatusEx(&statex);
    mgb_assert(succ, "GetPhysicallyInstalledSystemMemory failed");
    std::pair<size_t, size_t> ret;
    ret.first = statex.ullTotalPhys;
    ret.second = statex.ullAvailPhys;
    return ret;
}


#else // not WIN32

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#else
#include <sys/sysinfo.h>
#include <sched.h>
#endif

void sys::set_cpu_affinity(const std::vector<int> &cpuset) {
#if defined(__APPLE__) || !MGB_HAVE_THREAD
#pragma message("set_cpu_affinity not enabled on apple platform")
#else
    cpu_set_t mask;
    CPU_ZERO(&mask);
    auto nr = get_cpu_count();
    for (auto i: cpuset) {
        mgb_assert(i >= 0 && i < nr, "invalid CPU ID: nr_cpu=%d id=%d",
                nr, i);
        CPU_SET(i, &mask);
    }
    auto err = sched_setaffinity(0, sizeof(mask), &mask);
    if (err) {
        mgb_log_error("failed to sched_setaffinity: %s (error ignored)",
                strerror(errno));
    }
#endif
}

#ifdef MGB_EXTERN_API_MEMSTAT
extern "C" {
    void mgb_extern_api_memstat(size_t *tot, size_t *free);
}
#endif

std::pair<size_t, size_t> sys::get_ram_status_bytes() {
#ifdef MGB_EXTERN_API_MEMSTAT
    size_t tot, free;
    mgb_extern_api_memstat(&tot, &free);
    return {tot, free};
#elif defined(__APPLE__)
    static bool init_done;
    static std::mutex init_mtx;
    static mach_port_t host_port;
    static mach_msg_type_number_t host_size;
    static vm_size_t pagesize;

    {
        MGB_LOCK_GUARD(init_mtx);
        if (!init_done) {
            host_port = mach_host_self();
            host_size = sizeof(vm_statistics_data_t) / sizeof(integer_t);
            host_page_size(host_port, &pagesize);
            init_done = true;
        }
    }

    vm_statistics_data_t vm_stat;

    auto err = host_statistics(host_port, HOST_VM_INFO, (host_info_t)&vm_stat,
            &host_size);
    mgb_assert(err == KERN_SUCCESS);

    /* Stats in bytes */
    size_t mem_used = (vm_stat.active_count +
                       vm_stat.inactive_count +
                       vm_stat.wire_count) * pagesize;
    size_t mem_free = vm_stat.free_count * pagesize;
    return {mem_used + mem_free, mem_free};
#else
    struct sysinfo info;
    auto err = sysinfo(&info);
    mgb_assert(!err);
    std::pair<size_t, size_t> ret;
    ret.first = info.totalram * info.mem_unit;
    ret.second = (info.freeram + info.bufferram) * info.mem_unit;
    return ret;
#endif
}
#endif // WIN32

#if !MGB_BUILD_SLIM_SERVING && defined(__linux)
#include <unistd.h>
bool sys::stderr_ansi_color() {
    static bool ret = isatty(fileno(stderr));
    return ret;
}
#else
bool sys::stderr_ansi_color() {
    return false;
}
#endif

#if MGB_BUILD_SLIM_SERVING || defined(ANDROID) || defined(WIN32) || \
        defined(IOS) || defined(__APPLE__)

#pragma message("sys functions disabled on unsupported platforms")

void sys::set_thread_name(const std::string &) {
}

std::string sys::get_thread_name(Maybe<std::thread::id>) {
    return "@";
}

namespace {
    class FakeTimedFuncInvoker final: public TimedFuncInvoker {
        ThinHashMap<FuncId, Func> m_func_registry;

        void set_fork_exec_impl(const ForkExecImpl &) override {
        }

        void fork_exec_impl_mainloop(const char *) override {
            mgb_throw(MegBrainError,
                    "fork_exec_impl_mainloop should not be called in "
                    "SLIM_SERVING build");
        }

        void register_func(FuncId id, const Func &func,
                const FuncInit &func_init = {}) override {
            auto ins = m_func_registry.emplace(id, func);
            mgb_assert(ins.second, "duplicated id: %zu", id);
        }

        Maybe<Result> invoke(FuncId id, const Param &param, double) override {
            return m_func_registry.at(id)(param);
        }

        void kill_worker() override {
        }
    };
}

TimedFuncInvoker& TimedFuncInvoker::ins() {
    static FakeTimedFuncInvoker ins;
    return ins;
}

#else

#include <condition_variable>
#include <future>
#include <chrono>

#if MGB_ENABLE_DEBUG_UTIL
#include <sstream>
#endif

#include <cstring>

#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>

#if MGB_CUDA
#include <nvToolsExtCudaRt.h>
#endif

#define CHECK_SYS_ERR(_s) do { \
    if ((_s) < 0) { \
        auto _msg = ssprintf("%s failed: %s", #_s, strerror(errno)); \
        mgb_log_error("%s", _msg.c_str()); \
        mgb_throw_raw(SystemError{_msg}); \
    } \
} while(0)

namespace {
#if MGB_ENABLE_DEBUG_UTIL
ThinHashMap<std::thread::id, std::string> thread_name_map;
std::mutex thread_name_map_lock;
#endif
}  // anonymous namespace

void sys::set_thread_name(const std::string &name) {
#if MGB_ENABLE_DEBUG_UTIL
    MGB_LOCK_GUARD(thread_name_map_lock);
    thread_name_map[std::this_thread::get_id()] = name;

#if MGB_CUDA && MGB_ENABLE_DEBUG_UTIL
     nvtxNameOsThread(pthread_self(), name.c_str());
#endif

    auto ptr = name.c_str();
    for (; ; ) {
        auto ret = pthread_setname_np(pthread_self(), ptr);
        if (ret == ERANGE) {
            ++ ptr;
            continue;
        }
        mgb_assert(!ret, "failed to set thread name to %s: %s", name.c_str(),
                strerror(ret));
        break;
    }
#endif
}

std::string sys::get_thread_name(Maybe<std::thread::id> tid_) {
#if MGB_ENABLE_DEBUG_UTIL
    MGB_LOCK_GUARD(thread_name_map_lock);
    auto tid = tid_.val_with_default(std::this_thread::get_id());
    auto iter = thread_name_map.find(tid);
    if (iter == thread_name_map.end()) {
        std::ostringstream ostr;
        ostr << "unnamed_thread(" << tid << ")";
        return ostr.str();
    }
    return iter->second;
#else
    return "";
#endif
}

namespace {

class TimedFuncInvokerImpl final: public TimedFuncInvoker {
    /*
     * server-client protocol:
     *
     * server is the main megbrain process which calls invoke()
     *
     * client is the worker process that executes the function and may get
     * killed
     *
     * s: hello: rand uint32
     * c: hello + 1
     *
     * while true:
     *      s: func id, func arg len <size_t>, func arg
     *      c: init_done<uint8:1>, err<bool>, func result len <size_t>,
     *         func result; if error happens, err would be true and result is
     *         the error message
     */
    struct FuncRegistry {
        Func func;
        FuncInit init;

        Result direct_call(const Param &param) const {
            if (init)
                init(param);
            return func(param);
        }
    };
    static constexpr uint8_t INIT_DONE_FLAG = 23;
    ForkExecImpl m_fork_exec_impl;
    pid_t m_worker_pid = 0;
    int m_sock_fd = 0, m_peer_fd = 0, m_sock_name_cnt = 0;
    ThinHashMap<FuncId, FuncRegistry> m_func_registry;

    bool m_watcher_should_stop = false;
    std::condition_variable m_watcher_stop_cv;
    std::mutex m_watcher_stop_mtx, m_global_mtx;

    void clear_sock_fd() {
        if (m_peer_fd)
            close(m_peer_fd);
        if (m_sock_fd && m_sock_fd != m_peer_fd)
            close(m_sock_fd);
        m_sock_fd = m_peer_fd = 0;
    }

    void set_fork_exec_impl(const ForkExecImpl &impl) override {
        mgb_assert(!m_fork_exec_impl);
        m_fork_exec_impl = impl;
    }

    //! create an abstract AF_UNIX socket and bind to it
    void create_sock_and_bind(const char *name,
            int(*do_bind)(int, const sockaddr*, socklen_t)) {
        clear_sock_fd();

        m_sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        CHECK_SYS_ERR(m_sock_fd);

		struct sockaddr_un addr;
		addr.sun_family = AF_UNIX;
        addr.sun_path[0] = 0;
        auto name_len = strlen(name);
        mgb_assert(name_len < sizeof(addr.sun_path) - 1);
        strcpy(addr.sun_path + 1, name);
        auto len = sizeof(addr.sun_family) + name_len;
		CHECK_SYS_ERR(do_bind(m_sock_fd, (struct sockaddr *)&addr, len));
    }

    //! read from m_peer_fd and return whether success
    bool read(void *dest_, size_t size, bool throw_on_err = true) {
        auto dest = static_cast<uint8_t*>(dest_);
        while (size) {
            auto cur_recv = recv(m_peer_fd, dest, size, 0);
            if (!cur_recv && errno == EINTR)
                continue;
            if (!throw_on_err) {
                if (cur_recv <= 0)
                    return false;
            } else {
                CHECK_SYS_ERR(cur_recv);
            }
            mgb_assert(cur_recv > 0);
            dest += cur_recv;
            size -= cur_recv;
        }
        return true;
    }

    void write(const void *src_, size_t size) {
        auto src = static_cast<const uint8_t*>(src_);
        while (size) {
            auto cur_send = send(m_peer_fd, src, size, 0);
            CHECK_SYS_ERR(cur_send);
            mgb_assert(cur_send > 0);
            src += cur_send;
            size -= cur_send;
        }
    }

    template<class T>
    T read_pod() {
        static_assert(std::is_pod<T>::value, "can only read POD");
        T ret;
        read(&ret, sizeof(T));
        return ret;
    }

    template<class T>
    void write_pod(T val) {
        static_assert(std::is_pod<T>::value, "can only write POD");
        write(&val, sizeof(T));
    }

	void fork_exec_impl_mainloop(const char *arg) override {
        CHECK_SYS_ERR(prctl(PR_SET_PDEATHSIG, SIGKILL));

        create_sock_and_bind(arg, ::connect);
        m_peer_fd = m_sock_fd;

        // hello and handshake
        write_pod<uint32_t>(read_pod<uint32_t>() + 1);

        std::vector<uint8_t> param_buf;

        for (; ; ) {
            auto func_id = read_pod<FuncId>();
            auto param_size = read_pod<size_t>();
            param_buf.resize(param_size);
            read(param_buf.data(), param_size);

            bool init_done_written = false;

            bool err = false;
            Result res;
            auto setup_err = [&](const char *msg) {
                err = true;
                res.size = strlen(msg);
                res.data = std::make_unique<uint8_t[]>(res.size);
                memcpy(res.data.get(), msg, res.size);
            };
            MGB_MARK_USED_VAR(setup_err);
            Param func_param{param_size, param_buf.data()};
            MGB_TRY {
                auto &&entry = m_func_registry.at(func_id);
                if (entry.init) {
                    entry.init(func_param);
                }
                write_pod(INIT_DONE_FLAG);
                init_done_written = true;

                res = entry.func(func_param);
            } MGB_CATCH(std::exception &exc, {
                setup_err(exc.what());
            }) MGB_CATCH(..., {
                setup_err("unknown error");
            });
            if (!init_done_written) {
                write_pod(INIT_DONE_FLAG);
            }
            write_pod(err);
            write_pod(res.size);
            write(res.data.get(), res.size);
        }
	}

    void register_func(FuncId id,
            const Func &func, const FuncInit &init) override {
        mgb_assert(func);
        auto ins = m_func_registry.emplace(id, FuncRegistry{func, init});
        mgb_assert(ins.second, "duplicated id: %zu", id);
    }

    //! return whether worker is alive
    bool check_worker_alive() {
        if (m_worker_pid) {
            auto wait_ret = waitpid(m_worker_pid, nullptr, WNOHANG);
            CHECK_SYS_ERR(wait_ret);
            if (!wait_ret)
                return true;
        }
        return false;
    }

    //! start worker if it is not alive
    void ensure_worker_alive() {
        if (check_worker_alive())
            return;

        auto name = ssprintf("megbrain/%d/TimedFuncInvoker/%d",
                getpid(), m_sock_name_cnt ++);
        mgb_log_debug("start worker process on socket %s", name.c_str());

        create_sock_and_bind(name.c_str(), ::bind);
        CHECK_SYS_ERR(listen(m_sock_fd, 1));

        m_worker_pid = m_fork_exec_impl(name.c_str());
        mgb_assert(m_worker_pid > 0);
        m_peer_fd = accept(m_sock_fd, nullptr, nullptr);
        CHECK_SYS_ERR(m_peer_fd);

        uint32_t hello = time(nullptr);
        write_pod(hello);
        mgb_assert(read_pod<uint32_t>() == hello + 1);
    }

    Maybe<Result> invoke(
            FuncId id, const Param &param, double timeout) override {
        MGB_LOCK_GUARD(m_global_mtx);
        mgb_assert(timeout >= 0);
        auto iter = m_func_registry.find(id);
        mgb_assert(iter != m_func_registry.end(), "id %zu does not exist", id);
        if (!timeout && !check_worker_alive())
            return iter->second.direct_call(param);

        if (!m_fork_exec_impl) {
            mgb_log_warn("timeout is set, but no fork_exec_impl not given; "
                    "timeout would be ignored");
            return iter->second.direct_call(param);
        }

        // start worker and write init param; reading init_done sometimes fails
        // with connection reset, so we retry for some times
        constexpr int MAX_TRY = 5;
        for (int cur_try = 0; cur_try < MAX_TRY; ++ cur_try)  {
            ensure_worker_alive();
            write_pod(id);
            write_pod(param.size);
            write(param.data, param.size);
            std::remove_cv_t<decltype(INIT_DONE_FLAG)> init_done;
            if (!read(&init_done, sizeof(init_done), false)) {
                mgb_assert(cur_try < MAX_TRY - 1,
                        "can not read init_done flag");
                kill_worker();
                continue;
            }
            mgb_assert(init_done == INIT_DONE_FLAG);
            break;
        }
        m_watcher_should_stop = false;

        std::future<bool> watcher;
        if (timeout) {
            watcher = std::async(std::launch::async,
                    &TimedFuncInvokerImpl::watcher_impl, this, timeout);
        }

        // stop watcher, return whether worker killed by watcher
        auto stop_watcher = [&]() {
            if (!timeout)
                return false;

            {
                MGB_LOCK_GUARD(m_watcher_stop_mtx);
                m_watcher_should_stop = true;
                m_watcher_stop_cv.notify_all();
            }
            return watcher.get();
        };

        auto read_safe = [&](void *dest, size_t size) {
            if (!read(dest, size, false)) {
                if (!stop_watcher())
                    kill_worker();
                return false;
            }
            return true;
        };

        bool err;
        Result res;
        if (!read_safe(&err, sizeof(bool)) ||
                !read_safe(&res.size, sizeof(size_t)))
            return None;
        res.data = std::make_unique<uint8_t[]>(res.size + 1);
        if (!read_safe(res.data.get(), res.size))
            return None;
        if (stop_watcher())
            return None;
        res.data[res.size] = 0;
        if (err) {
            mgb_throw_raw(RemoteError{ssprintf(
                        "worker caught exception; what(): %s",
                        res.data.get())});
        }
        return {std::move(res)};
    }

    //! return whether kill has been issued
    bool watcher_impl(double timeout) {
        using namespace std::chrono;
        microseconds timeout_due{static_cast<uint64_t>(timeout * 1e6)};
        auto start = high_resolution_clock::now(),
             end = start + timeout_due;
        for (; ; ) {
            std::unique_lock<std::mutex> lk(m_watcher_stop_mtx);
            m_watcher_stop_cv.wait_until(lk, end);

            if (m_watcher_should_stop)
                return false;

            if (high_resolution_clock::now() >= end) {
                kill_worker();
                return true;
            }
        }
    }

    void kill_worker() override {
        if (m_worker_pid) {
            CHECK_SYS_ERR(kill(m_worker_pid, SIGKILL));
            auto w = waitpid(m_worker_pid, nullptr, 0);
            CHECK_SYS_ERR(w);
            mgb_assert(w == m_worker_pid);
            m_worker_pid = 0;
            clear_sock_fd();
        }
    }

    public:

        ~TimedFuncInvokerImpl() {
            std::exception_ptr pexc;
            MGB_TRY {
                MGB_TRY {
                    kill_worker();
                } MGB_CATCH_ALL_EXCEPTION("kill worker in ~TimedFuncInvokerImpl",
                        pexc);
            } MGB_CATCH(..., {});
            clear_sock_fd();
        }

};

} // anonymous namespace

TimedFuncInvoker& TimedFuncInvoker::ins() {
    static TimedFuncInvokerImpl impl;
    return impl;
}

std::unique_ptr<TimedFuncInvoker, TimedFuncInvoker::Del>
TimedFuncInvoker::make_test_ins() {
    return std::unique_ptr<TimedFuncInvoker, Del>{new TimedFuncInvokerImpl};
}

#undef CHECK_SYS_ERR

#endif // MGB_BUILD_SLIM_SERVING || defined(ANDROID)

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
