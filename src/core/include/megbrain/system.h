/**
 * \file src/core/include/megbrain/system.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thin/function.h"

#include <vector>
#include <string>
#include <memory>
#include <thread>

#include <cstring>

namespace mgb {
namespace sys {

    //! set name of caller thread
    void set_thread_name(const std::string &name);

    /*!
     * \brief get name of of given thread
     * \param tid thread id, or None to for the caller thread
     */
    std::string get_thread_name(Maybe<std::thread::id> tid = None);

    //! get number of CPU cores on this system
    int get_cpu_count();

    //! set cpu affinity for caller thread
    void set_cpu_affinity(const std::vector<int>& cpuset);

    //! whether stderr supports ansi color code
    bool stderr_ansi_color();

    //! get total ram and free ram in bytes
    std::pair<size_t, size_t> get_ram_status_bytes();

    /*!
     * \brief invoke a function with time limit
     *
     * This class should be accessed via the singleton ins().
     *
     * It is currently used to implement algorithm profiling because:
     *
     *      1. Some algos may be much slower (sometimes even more than 1000x)
     *         than others. Therefore we want to set a time limit so current
     *         algo can take no longer best known time.
     *      2. There is no portable and elegant way to interrupt an asynchronous
     *         function. So here we proceed by invoking the function in a child
     *         process and kill the whole process on timeout.
     *      3. We use fork-exec to launch the child process rather than using a
     *         simple fork because some device drivers (e.g. CUDA) would be
     *         broken if we fork without exec.
     *
     * For SDK developers (i.e. MegBrain users):
     *
     *      1. TimedFuncInvoker is currently only implemented for linux
     *         platforms when MGB_BUILD_SLIM_SERVING is disabled.
     *      2. You need to implement a fork-exec entry point and call
     *         TimedFuncInvoker::ins().set_fork_exec_impl() to setup the system.
     *      3. An example implementation is available in the python module.
     *
     * For algorithm profiling implementations (i.e. MegBrain developers):
     *
     *      1. Register the function to be profiled via register_func(). Use
     *         invoke() to call it with a timeout.
     *      2. You may need AlgoChooserProfileCache to save the profiling
     *         result.
     *
     */
    class TimedFuncInvoker: public NonCopyableObj {
        friend class TimedFuncInvokerTest;

        struct Del {
            void operator () (TimedFuncInvoker *p) {
                delete p;
            }
        };

        //! make an instance for test purpose
        static std::unique_ptr<TimedFuncInvoker, Del> make_test_ins();

        protected:
            virtual ~TimedFuncInvoker() = default;

        public:
            struct Result {
                size_t size = 0;
                std::unique_ptr<uint8_t[]> data;

                template<typename T>
                static Result from_pod(const T &val) {
                    Result ret{sizeof(T),
                        std::make_unique<uint8_t[]>(sizeof(T))};
                    memcpy(ret.data.get(), &val, sizeof(T));
                    return ret;
                }

                template<typename T>
                const T& as_single_pod() const {
                    static_assert(is_location_invariant<T>::value, "bad type");
                    mgb_assert(sizeof(T) == size);
                    return *reinterpret_cast<const T*>(data.get());
                }
            };
            struct Param {
                size_t size = 0;
                const uint8_t *data = nullptr;

                // param is non-const ref to ensure caller has ownership; it
                // would not be modified
                template<typename T>
                static Param from_pod(T &val) {
                    return {sizeof(T), reinterpret_cast<uint8_t*>(&val)};
                }

                template<typename T>
                const T& as_single_pod() const {
                    static_assert(is_location_invariant<T>::value, "bad type");
                    mgb_assert(sizeof(T) == size);
                    return *reinterpret_cast<const T*>(data);
                }
            };
            //! exception thrown by invoke()
            class RemoteError final: public MegBrainError {
                public:
                    using MegBrainError::MegBrainError;
            };

            using Func = thin_function<Result(const Param &param)>;
            using FuncInit = thin_function<void(const Param &param)>;
            using FuncId = size_t;

            /*!
             * \brief call fork() and exec(), and pass *arg* to the child
             * process, and the child process should pass *arg* back to
             * fork_exec_impl_mainloop()
             *
             * \param arg a null-terminated string argument
             * \return child process PID
             */
            using ForkExecImpl = thin_function<int(const std::string &arg)>;

            /*!
             * \brief set the function to implement fork-exec
             *
             * ForkExecImpl can not be implemented by TimedFuncInvoker because
             * it does not know the entry point of the compiled ELF.
             *
             * This method must be called from this server process, before any
             * call to invoke()
             */
            virtual void set_fork_exec_impl(const ForkExecImpl &impl) = 0;

            /*!
             * \brief to be called in the child process by ForkExecImpl
             *      registered by set_fork_exec_impl()
             *
             * \param arg the argument passed to ForkExecImpl
             */
            [[noreturn]] virtual void fork_exec_impl_mainloop(
                    const char *arg) = 0;

            /*!
             * \brief register a function that can be invoked
             *
             * This method must be called both from the server process and from
             * the client process. It is usually called during global setup.
             *
             * \param func the function associated with \p id; its execution
             *      time can not exceed given timeout
             * \param func_init an initializer whose time would not be counted
             *      in the timeout setting
             */
            virtual void register_func(FuncId id, const Func &func,
                    const FuncInit &func_init = {}) = 0;

            /*!
             * \brief invoke a function with given timeout
             *
             * This method must be called from the server process (i.e. main
             * mebrain process). This method is thread-safe.
             *
             * \param timeout timeout in seconds; if it is 0, timeout is
             *      disabled, and if worker not started yet, the function is
             *      invoked inplace
             * \return the return value if function finishes within given
             *      timeout; if the function could not finish in time, None
             *      would be returned.
             * \exception RemoteError thrown when function fails on the worker(
             *      function throws exception, not due to timeout); the error
             *      message would be forwarded to RemoteError::what()
             */
            virtual Maybe<Result> invoke(FuncId id, const Param &param,
                    double timeout) = 0;

            /*!
             * \brief kill the worker process
             */
            virtual void kill_worker() = 0;

            //! global unique instance
            static TimedFuncInvoker& ins();
    };

} // namespace sys
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

