/**
 * \file src/core/impl/utils/debug.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/debug.h"
#include "megdnn/tensor_iter.h"
#include <cmath>
#include <cerrno>

using namespace mgb;
using namespace debug;


#if MGB_ENABLE_DEBUG_UTIL

#include "megbrain/utils/metahelper.h"
#include "megbrain/common.h"

#include "megbrain/utils/thin/function.h"
#include <regex>
#include <cstring>
#include <cctype>
#include <cstdio>
#include <cstdlib>

#if MGB_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#ifdef __ANDROID__
#include <unwind.h>
#else
#include <execinfo.h>
#endif

#ifdef __ANDROID__
namespace {

struct AndroidBacktraceState {
    void** current;
    void** end;
};

static _Unwind_Reason_Code android_unwind_callback(struct _Unwind_Context* context,
                                           void* arg) {
    AndroidBacktraceState* state = static_cast<AndroidBacktraceState*>(arg);
    uintptr_t current_pc = _Unwind_GetIP(context);
    if(current_pc == nullptr)
        return _URC_NO_REASON;

    if (state->current == state->end) {
            return _URC_END_OF_STACK;
    } else {
            *state->current++ = reinterpret_cast<void*>(current_pc);
    }

    return _URC_NO_REASON;
}

size_t backtrace(void** buffer, size_t max) {
    AndroidBacktraceState state = {buffer, buffer + max};
    _Unwind_Backtrace(android_unwind_callback, &state);
    return state.current - buffer;
}

}  // anonymous namespace
#endif  // backtrace impl for __ANDROID__

namespace {

void throw_fork_cuda_exc() {
    mgb_throw(ForkAfterCudaError, "fork after cuda has been initialized");
}

struct MemmapEntry {
    uintptr_t low, high;
    std::string file;

    MemmapEntry(uint64_t low_, uint64_t high_, const char *file_):
        low(low_), high(high_), file(file_)
    {}
};

void get_mem_map(int pid, thin_function<void(
            uintptr_t, uintptr_t, const char*, const char*)> callback) {
    char fpath[64];
    if (pid)
        sprintf(fpath, "/proc/%d/maps", pid);
    else
        strcpy(fpath, "/proc/self/maps");
    FILE *fin = fopen(fpath, "r");
    mgb_assert(fin, "failed to open %s", fpath);
    char linebuf[512];
    while (fgets(linebuf, sizeof(linebuf), fin)) {
        uintptr_t begin, end;
        char perm[10], offset[20], dev[10], inode[20], path_mem[256], *path;
        int nr = sscanf(linebuf, "%zx-%zx %s %s %s %s %s",
                &begin, &end, perm, offset, dev, inode, path_mem);
        if (nr == 6)
            path = nullptr;
        else {
            mgb_assert(nr == 7, "failed to parse map line: %s", linebuf);
            path = path_mem;
        }
        callback(begin, end, perm, path);
    }
    fclose(fin);
}

class SigHandlerInit {
    static void death_handler(int signum) {
        char msg0[] = "megbrain is about to die abruptly; you can set "
            "MGB_WAIT_TERMINATE and rerun to wait for gdb attach";
        if (MGB_GETENV("MGB_WAIT_TERMINATE")) {
            fprintf(stderr, "megbrain is about to die abruptly; you can gdb "
                    "me at %d; wait for pressing enter\n",
                    static_cast<int>(getpid()));
            getchar();
        }
        if (signum == -1) {
            mgb_log_error("%s: std::terminate() called", msg0);
        } else {
            mgb_log_error("%s: caught deadly signal %d(%s)",
                    msg0,
                    signum, strsignal(signum));
        }
        std::string bp;
        debug::backtrace(2).fmt_to_str(bp);
        mgb_log_error("%s", bp.c_str());
        exit(EXIT_FAILURE);
    }

    public:
        static void init_for_segv() {
            struct sigaction action;
            memset(&action, 0, sizeof(action));
            action.sa_handler = &death_handler;
            sigaction(SIGSEGV, &action, nullptr);
            std::set_terminate([](){death_handler(-1);});
        }
};

#if MGB_CUDA
class CudaCheckOnFork {
    static int& flag() {
        static int ret = MGB_GETENV("MGB_THROW_ON_FORK") ? 2 : 1;
        return ret;
    }

    static void atfork_prepare() {
        if (flag() && !ScopedForkWarningSupress::supress()) {
            CUcontext ctx;
            if (cuCtxGetCurrent(&ctx) != CUDA_ERROR_NOT_INITIALIZED) {
                mgb_log_debug("It is dangerous to call fork() after cuda "
                        "context has been initialized; please ensure no cuda "
                        "methods is invoked in the child process. You can set "
                        "MGB_THROW_ON_FORK to find out where the fork() is "
                        "called.");

                if (flag() > 1) {
                    ForkAfterCudaError::throw_();
                }
            }
        }
    }

    public:
        static void set_flag(int f) {
            flag() = f;
        }

        static void init() {
            int err = pthread_atfork(&CudaCheckOnFork::atfork_prepare,
                    nullptr, nullptr);
            if (err) {
                mgb_throw(SystemError,
                        "failed to setup atfork handler: %s",
                        strerror(err));
            }
        }
};
#endif

class InitCaller {
    static InitCaller inst;

    InitCaller() {
        SigHandlerInit::init_for_segv();
#if MGB_CUDA
        CudaCheckOnFork::init();
#endif
    }
};
InitCaller InitCaller::inst;

} // anonymous namespace

void (*ForkAfterCudaError::throw_)() = throw_fork_cuda_exc;

std::atomic_size_t ScopedForkWarningSupress::sm_depth{0};

BacktraceResult mgb::debug::backtrace(int nr_exclude) {
    static bool thread_local recursive_call = false;
    if (recursive_call) {
        fprintf(stderr, "recursive call to backtrace()!\n");
        return {};
    }
    recursive_call = true;

    constexpr size_t MAX_DEPTH = 6;
    void *stack_mem[MAX_DEPTH];
    int depth = ::backtrace(stack_mem, MAX_DEPTH);
    auto stack = stack_mem;
    if (depth > nr_exclude) {
        depth -= nr_exclude;
        stack += nr_exclude;
    }

    static std::vector<MemmapEntry> memmap;
    if (memmap.empty()) {
        static std::mutex mtx;
        MGB_LOCK_GUARD(mtx);
        if (memmap.empty()) {
            get_mem_map(0, [&](uintptr_t lo, uintptr_t hi,
                        const char * /*perm*/, const char *fname){
                    if (fname && strlen(fname))
                    memmap.emplace_back(lo, hi, fname);
                    });
        }
    }
    BacktraceResult result;

    for (int i = 0; i < depth; ++ i) {
        const char* fname = nullptr;
        auto addr = reinterpret_cast<uintptr_t>(stack[i]);
        for (auto &&j: memmap)
            if (j.low <= addr && j.high >= addr) {
                // theoretically we should examine file content to find whether
                // it is a shared library; but who would name an executable with
                // .so ?
                if (j.file.find(".so") != std::string::npos)
                    addr -= j.low;

                fname = j.file.c_str();
                break;
            }
        result.stack.emplace_back(fname, addr);
    }

    recursive_call = false;
    return result;
}

void BacktraceResult::fmt_to_str(std::string &dst) {
    char addr[128];
    bool first = true;
    const char* prev_fname = nullptr;
    dst.append("bt:");
    for (auto &&i: stack) {
        sprintf(addr, "%zx", i.second);
        if (i.first != prev_fname || first) {
            if (!first)
                dst.append("}");
            if (i.first)
                dst.append(i.first);
            else
                dst.append("unknown");
            prev_fname = i.first;
            first = false;
            dst.append("{");
            dst.append(addr);
        } else {
            dst.append(",");
            dst.append(addr);
        }
    }
    dst.append("}");
}

void debug::set_fork_cuda_warning_flag(int flag) {
#if MGB_CUDA
    CudaCheckOnFork::set_flag(flag);
#endif
}

#endif // MGB_ENABLE_DEBUG_UTIL

namespace {

    bool good_float(float val) {
        return std::isfinite(val);
    }

    bool good_float(int) {
        return true;
    }

#if MGB_ENABLE_LOGGING
    // if not in MGB_ENABLE_LOGGING, num2str would become defined but not used
    template<typename T>
    std::string num2str(T val) {
        return std::to_string(val);
    }

    std::string num2str(float val) {
        union V {
            uint32_t i;
            float f;
        };
        auto ret = std::to_string(val);
        if (!good_float(val)) {
            V v;
            v.f = val;
            ret.append(" (0x");
            ret.append(ssprintf("%x", v.i));
            ret.append(")");
        }
        return ret;
    }
#endif

    template<typename ctype>
    Maybe<std::string> do_compare_tensor_value(
        const char *expr0, const char *expr1,
        const HostTensorND &v0, const HostTensorND &v1, float maxerr) {

        auto it0 = megdnn::tensor_iter<ctype>(v0.as_megdnn()).begin(),
             it1 = megdnn::tensor_iter<ctype>(v1.as_megdnn()).begin();
        for (size_t i = 0, it = v0.shape().total_nr_elems(); i < it; ++ i) {
            ctype iv0 = *it0, iv1 = *it1;
            double err = std::abs(iv0 - iv1) / std::max<double>(
                    1, std::min(std::abs(static_cast<double>(iv0)),
                        std::abs((static_cast<double>(iv1)))));
            if (!good_float(iv0) || !good_float(iv1) || err >= maxerr) {
                TensorShape idx_shp;
                idx_shp.ndim = v0.shape().ndim;
                std::copy(it0.idx(), it0.idx() + idx_shp.ndim, idx_shp.shape);
                return mgb_ssprintf_log(
                        "Unequal value\n"
                        "Value of: %s\n"
                        "  Actual: %s\n"
                        "Expected: %s\n"
                        "Which is: %s\n"
                        "At index: %s/%s\n"
                        "   error: %.6g",
                        expr1, num2str(iv1).c_str(),
                        expr0, num2str(iv0).c_str(),
                        idx_shp.to_string().c_str(),
                        v0.shape().to_string().c_str(),
                        err);
            }

            ++ it0;
            ++ it1;
        }
        return None;
    }

} // anonymous namespace

Maybe<std::string> debug::compare_tensor_value(
        const HostTensorND &v0, const char *expr0,
        const HostTensorND &v1, const char *expr1,
        float maxerr) {
    if (!v0.shape().eq_shape(v1.shape())) {
        return mgb_ssprintf_log(
                "Shape mismatch\n"
                "Value of: %s\n"
                "  Actual: %s\n"
                "Expected: %s\n"
                "Which is: %s",
                expr1, v1.shape().to_string().c_str(),
                expr0, v0.shape().to_string().c_str());
    }
    auto dtype = v0.layout().dtype;
    if (dtype != v1.layout().dtype) {
        return mgb_ssprintf_log(
                "Data type mismatch\n"
                "Value of: %s\n"
                "  Actual: %s\n"
                "Expected: %s\n"
                "Which is: %s",
                expr1, v1.layout().dtype.name(),
                expr0, v0.layout().dtype.name());
    }

    switch(dtype.enumv()) {
#define cb(_dt) case DTypeTrait<_dt>::enumv: \
        return do_compare_tensor_value<DTypeTrait<_dt>::ctype>( \
                expr0, expr1, v0, v1, maxerr);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
        default:
            mgb_throw(MegBrainError, "unhandled dtype: %s", dtype.name());
    }
}

std::string debug::dump_tensor(const HostTensorND& value, const std::string &name) {
    struct Header {
        uint32_t name_len;
        uint32_t dtype;
        uint32_t max_ndim;
        uint32_t shape[TensorShape::MAX_NDIM];
        char name[0];
    };
    mgb_assert(value.layout().is_contiguous());
    auto value_bytes = value.layout().span().dist_byte();
    std::string ret(name.size() +  value_bytes + sizeof(Header), '\0');
    auto header = reinterpret_cast<Header*>(&ret[0]);
    memset(header, 0, sizeof(Header));
    header->name_len = name.length();
    header->dtype = static_cast<uint32_t>(value.dtype().enumv());
    header->max_ndim = TensorShape::MAX_NDIM;
    for (size_t i = 0; i < value.layout().ndim; ++i) {
        header->shape[i] = value.layout()[i];
    }
    memcpy(header->name, name.c_str(), header->name_len);
    memcpy(header->name + name.size(), value.raw_ptr(), value_bytes);
    return ret;
}

void debug::write_to_file(const char* filename, const std::string& content,
                          const char* mode) {
    FILE* fout = fopen(filename, mode);
    mgb_throw_if(!fout, SystemError, "failed to open %s: %s", filename,
                 strerror(errno));
    auto nr = fwrite(content.data(), 1, content.size(), fout);
    mgb_throw_if(nr != content.size(), SystemError,
                 "failed to write to %s: num=%zu size=%zu %s", filename,
                 nr, content.size() ,strerror(errno));
    auto err = fclose(fout);
    mgb_throw_if(err, SystemError, "failed to close %s: %s", filename,
                 strerror(errno));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
