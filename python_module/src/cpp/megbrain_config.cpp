/**
 * \file python_module/src/cpp/megbrain_config.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "./megbrain_config.h"
#include "./python_helper.h"

#include "megbrain/graph/event.h"
#include "megbrain/utils/debug.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/serialization/opr_registry.h"

#include <set>

#include <dlfcn.h>

#if MGB_CUDA
#include <cuda.h>
#endif

using namespace mgb;

namespace {
    std::unordered_map<ComputingGraph*,
        SyncEventConnecter::ReceiverHandler>
        set_priority_on_opr_inserted_handle;
    std::mutex set_priority_on_opr_inserted_handle_mtx;

} // anonymous namespace

bool _config::set_comp_graph_option(
        CompGraph &cg, const std::string &name, int val_int) {

#define SET_CG_OPTION(name_chk) \
    do { \
        static_assert( \
                std::is_same<decltype(opt.name_chk), bool>::value || \
                std::is_same<decltype(opt.name_chk), uint8_t>::value || \
                std::is_same<decltype(opt.name_chk), int16_t>::value || \
                std::is_same<decltype(opt.name_chk), uint16_t>::value, \
                "not bool/int opt"); \
        if (name == #name_chk) { \
            auto ret = opt.name_chk; \
            opt.name_chk = val_int; \
            return ret; \
        } \
    } while(0)

    auto &&opt = cg.get().options();
    SET_CG_OPTION(seq_opt.enable_mem_plan_opt);
    SET_CG_OPTION(seq_opt.enable_mem_reuse_alloc);
    SET_CG_OPTION(seq_opt.enable_seq_comp_node_opt);
    SET_CG_OPTION(force_dynamic_alloc);
    SET_CG_OPTION(enable_grad_var_static_reshape);
    SET_CG_OPTION(async_exec_level);
    SET_CG_OPTION(graph_opt.jit);
    SET_CG_OPTION(graph_opt.tensorrt);
    SET_CG_OPTION(graph_opt_level);
    SET_CG_OPTION(var_sanity_check_first_run);
    SET_CG_OPTION(no_profiling_on_shape_change);
    SET_CG_OPTION(allocate_static_mem_after_graph_compile);
    SET_CG_OPTION(log_level);
    SET_CG_OPTION(enable_sublinear_memory_opt);
    SET_CG_OPTION(enable_var_mem_defragment);
    SET_CG_OPTION(eager_evaluation);
    SET_CG_OPTION(enable_memory_swap);
    throw MegBrainError(ssprintf(
                "invalid computing graph option name: %s", name.c_str()));
#undef SET_CG_OPTION
}

bool _config::comp_graph_is_eager(CompGraph &cg) {
    return cg.get().options().eager_evaluation;
}

void _config::add_extra_vardep(const SymbolVar &var, const SymbolVar &dep) {
    auto og = var.node()->owner_graph();
    mgb_assert(og == dep.node()->owner_graph());
    og->options().extra_vardeps[var.node()].push_back(dep.node());
}

void _config::begin_set_opr_priority(CompGraph& cg, int priority) {
    SyncEventConnecter::ReceiverHandler* handle;
    {
        MGB_LOCK_GUARD(set_priority_on_opr_inserted_handle_mtx);
        handle = &set_priority_on_opr_inserted_handle[&cg.get()];
    }
    mgb_assert(!*handle, "multiple calls to _begin_set_opr_priority()");

    auto on_opr_inserted = [priority](const cg::event::OprInserted& event) {
        if (!event.exc && priority) {
            int& pri = event.opr->node_prop().attribute().priority;
            if (!pri)
                pri = priority;
            else
                pri = std::min(pri, priority);
        }
    };
    *handle = cg.get().event().register_receiver<cg::event::OprInserted>(
            on_opr_inserted);
}

void _config::end_set_opr_priority(CompGraph &cg) {
    MGB_LOCK_GUARD(set_priority_on_opr_inserted_handle_mtx);
    auto nr = set_priority_on_opr_inserted_handle.erase(&cg.get());
    mgb_assert(nr, "end_set_opr_priority called "
            "before begin_set_opr_priority");
}

void _config::begin_set_exc_opr_tracker(CompGraph &cg, PyObject *tracker) {
    OprPyTracker::begin_set_tracker(cg.get(), tracker);
}

void _config::end_set_exc_opr_tracker(CompGraph &cg) {
    OprPyTracker::end_set_tracker(cg.get());
}

PyObject* _config::get_opr_tracker(CompGraph &cg, size_t var_id) {
    auto var = cg.get().find_var_by_id(var_id);
    if (!var)
        Py_RETURN_NONE;
    return OprPyTracker::get_tracker(var->owner_opr()).as_tuple();
}

void _config::set_opr_sublinear_memory_endpoint(const SymbolVar &var) {
    MGB_MARK_USED_VAR(var);
#if MGB_ENABLE_SUBLINEAR
    auto opr = var.node()->owner_opr();
    opr->owner_graph()->options().opr_attribute.sublinear_memory_endpoint.
        insert(opr);
#endif
}

void _config::set_fork_cuda_warning_flag(int flag) {
#if MGB_ENABLE_DEBUG_UTIL
    debug::set_fork_cuda_warning_flag(flag);
#else
    MGB_MARK_USED_VAR(flag);
#endif
}

bool _config::is_cuda_ctx_set() {
#if MGB_CUDA
    CUcontext ctx;
    return cuCtxGetCurrent(&ctx) == CUDA_SUCCESS && ctx;
#else
    return false;
#endif
}

std::string _config::get_cuda_gencode() {
#if MGB_CUDA
    std::set<std::string> used;
    int nr_dev;
    auto err = cudaGetDeviceCount(&nr_dev);
    if (err == cudaErrorNoDevice) {
        return {};
    }
    MGB_CUDA_CHECK(err);
    for (int i = 0; i < nr_dev; ++ i) {
        cudaDeviceProp prop;
        MGB_CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::string cur{std::to_string(prop.major)};
        cur += std::to_string(prop.minor);
        used.insert(cur);
    }

    std::string ret;
    for (auto &&i: used) {
        if (!ret.empty())
            ret.append(" ");
        ret.append(i);
    }
    return ret;
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}

namespace {
#if MGB_CUDA
    std::string get_loaded_shared_lib_path(const char* sl_name) {
        char path[PATH_MAX];
        auto handle = dlopen(sl_name,
                             RTLD_GLOBAL | RTLD_LAZY | RTLD_NOLOAD);
        mgb_assert(handle != nullptr, "%s", dlerror());
        mgb_assert(dlinfo(handle, RTLD_DI_ORIGIN, &path) != -1,
                   "%s", dlerror());
        return path;
    }
#endif
}

std::vector<std::string> _config::get_cuda_include_path() {
#if MGB_CUDA
    auto cuda_path = getenv("CUDA_BIN_PATH");
    if (cuda_path) {
        return std::vector<std::string>{cuda_path,
                                        std::string(cuda_path) + "/include"};
    } else {
        auto cuda_lib_path = get_loaded_shared_lib_path("libcudart.so");
        return {cuda_lib_path, cuda_lib_path + "/../",
                cuda_lib_path + "/../include"};
    }
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}

std::vector<std::string> _config::get_cuda_lib_path() {
#if MGB_CUDA
    auto cuda_path = getenv("CUDA_BIN_PATH");
    if (cuda_path) {
        return std::vector<std::string>{cuda_path,
                                        std::string(cuda_path) + "/lib64"};
    } else {
        auto cuda_lib_path = get_loaded_shared_lib_path("libcudart.so");
        return {cuda_lib_path};
    }
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}

int _config::get_cuda_version() {
#if MGB_CUDA
    int version;
    MGB_CUDA_CHECK(cudaRuntimeGetVersion(&version));
    return version;
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}

bool _config::is_compiled_with_cuda() {
#if MGB_CUDA
    return true;
#else
    return false;
#endif
}

void _config::load_opr_library(const char* self_path, const char* lib_path) {
    static bool self_global = false;
    static std::mutex self_global_mtx;
    {
        MGB_LOCK_GUARD(self_global_mtx);
        if (!self_global) {
            auto hdl = dlopen(self_path, RTLD_LAZY | RTLD_GLOBAL);
            mgb_assert(hdl, "failed to set mgb to global: %s", dlerror());
            self_global = true;
        }
    }
    if (lib_path) {
        auto hdl = dlopen(lib_path, RTLD_LAZY);
        mgb_assert(hdl, "failed to load libray %s: %s", lib_path, dlerror());
    }
}

std::vector<std::pair<uint64_t, std::string>> _config::dump_registered_oprs() {
#if MGB_ENABLE_DEBUG_UTIL
    return serialization::OprRegistry::dump_registries();
#else
    return {};
#endif
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
