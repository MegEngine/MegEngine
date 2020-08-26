/**
 * \file src/jit/impl/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#if MGB_JIT

#include "megbrain/utils/debug.h"
#include "megbrain/jit/utils.h"

#include <atomic>

#ifdef __linux__
#include <dlfcn.h>
#include <ftw.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif  // __linux__

using namespace mgb;
using namespace jit;

/* ====================== str_util ====================== */

void str_util::replace_all_pairs_inplace(
        std::string& text,
        const std::vector<std::pair<std::string, std::string>>& replace) {
    using str = std::string;
    auto repl_one = [&text](const str& from, const str& to) {
        mgb_assert(!from.empty());
        size_t pos = 0;
        while ((pos = text.find(from, pos)) != str::npos) {
            text.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    for (auto&& i : replace) {
        repl_one(i.first, i.second);
    }
}

/* ====================== ExecutableHelper ====================== */

bool ExecutableHelper::keep_interm() {
    static bool ret = MGB_GETENV("MGB_JIT_KEEP_INTERM");
    return ret;
}

namespace {

#ifdef __linux__

class ExecutableHelperImpl final : public ExecutableHelper {
    bool m_workdir_need_rm = false;

    //! workdir setting, end with /
    std::string m_workdir;

    //! execute command and check if exit code is zero
    static void check_exec(const std::string& cmd) {
#if MGB_ENABLE_DEBUG_UTIL
        debug::ScopedForkWarningSupress no_fork_warning;
#endif
        std::string out;
        std::array<char, 128> buffer;
        FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
        mgb_throw_if(!pipe, SystemError, "popen() for cmd %s failed: %s",
                     cmd.c_str(), strerror(errno));
        std::unique_ptr<FILE, int (*)(FILE*)> pipe_close{pipe, ::pclose};
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            out += buffer.data();
        }
        pipe_close.release();
        int ret = pclose(pipe);
        mgb_throw_if(ret, SystemError,
                     "command %s failed: return code=%d; captured output:\n%s",
                     cmd.c_str(), ret, out.c_str());
    }

public:
    ExecutableHelperImpl() {
        if (auto set = MGB_GETENV("MGB_JIT_WORKDIR")) {
            struct stat sb;
            if (!(stat(set, &sb) == 0 && S_ISDIR(sb.st_mode))) {
                int err = mkdir(set, 0700);
                mgb_throw_if(err, SystemError, "failed to create dir %s: %s",
                             set, strerror(errno));
                m_workdir_need_rm = true;
            }
            m_workdir = set;
        } else {
            char name[] = "/tmp/mgbjit-XXXXXX";
            auto ptr = mkdtemp(name);
            mgb_throw_if(!ptr, SystemError, "failed to create temp dir: %s",
                         strerror(errno));
            m_workdir = ptr;
            m_workdir_need_rm = true;
        }
        struct stat sb;
        mgb_throw_if(
                !(stat(m_workdir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)),
                SystemError, "%s is not a dir", m_workdir.c_str());
        mgb_log("use JIT workdir: %s", m_workdir.c_str());
        if (m_workdir.back() != '/')
            m_workdir.append("/");
    }

    ~ExecutableHelperImpl() {
        if (!m_workdir_need_rm || keep_interm())
            return;

        // remove work dir
        auto cb_rm = [](const char* fpath, const struct stat* sb, int typeflag,
                        struct FTW* ftwbuf) -> int {
            int err = ::remove(fpath);
            if (err) {
                mgb_log_error("failed to remove %s: %s", fpath,
                              strerror(errno));
            } else {
                mgb_log_debug("removed temp file in workdir: %s", fpath);
            }
            return err;
        };
        int err = nftw(m_workdir.c_str(), cb_rm, 64, FTW_DEPTH | FTW_PHYS);
        if (err) {
            mgb_log_error("failed to cleanup workdir %s", m_workdir.c_str());
        }
    }

    void* load_lib(const std::string& name) override {
        auto ret = dlopen(realpath(name).c_str(), RTLD_LAZY | RTLD_LOCAL);
        mgb_throw_if(!ret, SystemError, "failed to load library %s: %s",
                     name.c_str(), dlerror());
        return ret;
    }

    void* resolve_func(void* handle, const std::string& func_name) override {
        auto ret = dlsym(handle, func_name.c_str());
        mgb_throw_if(!ret, SystemError, "failed to resolve %s: %s",
                     func_name.c_str(), dlerror());
        return ret;
    }

    void unload_lib(void* handle) override {
        if (handle) {
            struct link_map* lmap;
            std::string path;
            bool path_good;
            if (dlinfo(handle, RTLD_DI_LINKMAP, &lmap)) {
                path_good = false;
                path = ssprintf("<RTLD_DI_ORIGIN failed: %s>", dlerror());
            } else {
                path_good = true;
                path = lmap->l_name;
            }
            if (dlclose(handle)) {
                mgb_log_error("failed to close %s: %s", path.c_str(),
                              dlerror());
            }
            if (path_good) {
                auto h1 = dlopen(path.c_str(), RTLD_NOLOAD | RTLD_LOCAL);
                if (h1) {
                    dlclose(h1);
                    mgb_log_warn("library %s is not totally released",
                                 path.c_str());
                }
            }
        }
    }

    std::string compile_cpp_source_secondary(const char* source,
                                             const char* out_name) override {
        std::string uniq_name{out_name};
        uniq_name.append("-");
        uniq_name.append(std::to_string(
                XXHash{}.update(source, strlen(source)).digest()));
        auto src_name = uniq_name + ".cpp", obj_name = uniq_name + ".o";
        write_file(src_name, source);
        check_exec(ssprintf("g++ -O2 -fPIC -std=c++11 '%s' -o '%s' -c",
                            realpath(src_name).c_str(),
                            realpath(obj_name).c_str()));
        return obj_name;
    }

    void link(const SmallVector<std::string>& inp_names,
              const std::string& out_name) override {
        std::string cmd{"g++ -shared -std=c++11 -o '"};
        cmd.append(realpath(out_name));
        cmd.append("'");
        for (auto&& i : inp_names) {
            cmd.append(" '");
            cmd.append(realpath(i));
            cmd.append("'");
        }
        check_exec(cmd);
    }

    std::string realpath(const std::string& name) override {
        mgb_assert(name.find('/') == std::string::npos);
        return m_workdir + name;
    }

    void remove(const std::string& name) override {
        int err = unlink(realpath(name).c_str());
        mgb_throw_if(err, SystemError, "failed to unlink %s: %s", name.c_str(),
                     strerror(errno));
    }
};

#endif  // __linux__

}  // anonymous namespace

void ExecutableHelper::write_file(const std::string& name,
                                  const std::string& data) {
    auto full_name = realpath(name);
    FILE* fptr = fopen(full_name.c_str(), "wb");
    mgb_throw_if(!fptr, SystemError, "failed to open %s: %s", full_name.c_str(),
                 strerror(errno));
    std::unique_ptr<FILE, int (*)(FILE*)> fptr_close{fptr, ::fclose};
    auto done = fwrite(data.data(), 1, data.size(), fptr);
    mgb_throw_if(done != data.size(), SystemError,
                 "failed to write file: req=%zu written=%zu: %s", data.size(),
                 done, strerror(errno));
    fptr_close.release();
    int err = fclose(fptr);
    mgb_throw_if(err, SystemError, "failed to close file: %s", strerror(errno));
}

ExecutableHelper& ::ExecutableHelper::get() {
    static ExecutableHelperImpl inst;
    return inst;
}

std::string jit::next_kernel_name() {
    static std::atomic_uint_fast64_t cnt;
    return "fusion" + std::to_string(cnt.fetch_add(1));
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
