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
#include <fstream>
#include <string>
#include <sstream>

#ifdef WIN32
#include <io.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/mm_handler.h"
#endif

#if MGB_CUDA
#include <cuda.h>
#endif

#ifdef WIN32
#define F_OK 0
#define RTLD_LAZY 0
#define RTLD_GLOBAL 0
#define RTLD_NOLOAD 0
#define access(a, b) false
#define SPLITER ';'
#define ENV_PATH "Path"
#define NVCC_EXE "nvcc.exe"
static void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibrary(file));
}

static void* dlerror() {
    const char* errmsg = "dlerror not aviable in windows";
    return const_cast<char*>(errmsg);
}

static void* dlsym(void* handle, const char* name) {
    FARPROC symbol = GetProcAddress((HMODULE)handle, name);
    return reinterpret_cast<void*>(symbol);
}

static int check_file_exist(const char* path, int mode) {
    return _access(path, mode);
}
#else
#define SPLITER ':'
#define ENV_PATH "PATH"
#define NVCC_EXE "nvcc"
static int check_file_exist(const char* path, int mode) {
    return access(path, mode);
}
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
                std::is_same<decltype(opt.name_chk), uint16_t>::value || \
                std::is_same<decltype(opt.name_chk), int32_t>::value, \
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
    SET_CG_OPTION(allreduce_pack_max_size);
    SET_CG_OPTION(allreduce_pack_ignore_first);
    SET_CG_OPTION(var_sanity_check_first_run);
    SET_CG_OPTION(no_profiling_on_shape_change);
    SET_CG_OPTION(allocate_static_mem_after_graph_compile);
    SET_CG_OPTION(log_level);
    SET_CG_OPTION(enable_sublinear_memory_opt);
    SET_CG_OPTION(sublinear_mem_config.lb_memory);
    SET_CG_OPTION(sublinear_mem_config.genetic_nr_iter);
    SET_CG_OPTION(sublinear_mem_config.genetic_pool_size);
    SET_CG_OPTION(sublinear_mem_config.thresh_nr_try);
    SET_CG_OPTION(sublinear_mem_config.num_worker);
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

std::string find_content_in_file(const std::string& file_name,
                                 const std::string& content) {
    std::ifstream fin(file_name.c_str());
    std::string read_str;
    while (std::getline(fin, read_str)) {
        auto idx = read_str.find(content);
        if (idx != std::string::npos) {
            fin.close();
            return read_str.substr(idx);
        }
    }
    fin.close();
    return {};
}

std::vector<std::string> split_env(const char* env) {
    std::string e(env);
    std::istringstream stream(e);
    std::vector<std::string> ret;
    std::string path;
    while (std::getline(stream, path, SPLITER)) {
        ret.emplace_back(path);
    }
    return ret;
}

//! this function will find file_name in each path in envs. It accepts add
//! intermediate path between env and file_name
std::string find_file_in_envs_with_intmd(
        const std::vector<std::string>& envs, const std::string& file_name,
        const std::vector<std::string>& itmedias = {}) {
    for (auto&& env : envs) {
        auto ret = getenv(env.c_str());
        if (ret) {
            for (auto&& path : split_env(ret)) {
                auto file_path = std::string(path) + "/" + file_name;
                if (!check_file_exist(file_path.c_str(), F_OK)) {
                    return file_path;
                }
                if (!itmedias.empty()) {
                    for (auto&& inter_path : itmedias) {
                        file_path = std::string(path) + "/" + inter_path + "/" +
                                    file_name;
                        if (!check_file_exist(file_path.c_str(), F_OK)) {
                            return file_path;
                        }
                    }
                }
            }
        }
    }
    return std::string{};
}

std::string get_nvcc_root_path() {
    auto nvcc_root_path = find_file_in_envs_with_intmd({ENV_PATH}, NVCC_EXE);
    if (nvcc_root_path.empty()) {
        mgb_throw(MegBrainError,
                  "nvcc not found. Add your nvcc to your environment Path");
    } else {
        auto idx = nvcc_root_path.rfind('/');
        return nvcc_root_path.substr(0, idx + 1);
    }
}

size_t get_local_cuda_version() {
    auto nvcc_root_path = get_nvcc_root_path();
    auto ver_path = nvcc_root_path + "../version.txt";
    if (check_file_exist(ver_path.c_str(), F_OK)) {
        mgb_throw(MegBrainError, "No such file : %s\n", ver_path.c_str());
    }
    auto str_cuda_version = find_content_in_file(ver_path, "CUDA Version");
    if (str_cuda_version.empty()) {
        mgb_throw(MegBrainError, "can not read version information from : %s\n",
                  ver_path.c_str());
    }
    size_t cuda_major = 0;
    size_t cuda_minor = 0;
    sscanf(str_cuda_version.c_str(), "CUDA Version %zu.%zu,", &cuda_major,
           &cuda_minor);
    return cuda_major * 1000 + cuda_minor * 10;
}

void check_cudnn_existence() {
    auto cudnn_header_path = find_file_in_envs_with_intmd(
            {"PC_CUDNN_INCLUDE_DIRS", "CUDNN_ROOT_DIR", "CUDA_TOOLKIT_INCLUDE",
             "CUDNN_LIBRARY", "CUDA_PATH"},
            "cudnn.h", {"../include", "include"});
    if (cudnn_header_path.empty()) {
        mgb_log_warn(
                "cudnn.h not found. Please make sure cudnn install at "
                "${CUDNN_ROOT_DIR}");
    } else {  // check cudnn lib exist
        auto str_cudnn_major =
                find_content_in_file(cudnn_header_path, "#define CUDNN_MAJOR");
        auto str_cudnn_minor =
                find_content_in_file(cudnn_header_path, "#define CUDNN_MINOR");
        auto str_cudnn_patch = find_content_in_file(cudnn_header_path,
                                                    "#define CUDNN_PATCHLEVEL");

        if (str_cudnn_major.empty() || str_cudnn_minor.empty() ||
            str_cudnn_patch.empty()) {
            mgb_log_warn(
                    "can not find cudnn version information in %s.\n You may "
                    "Update cudnn\n",
                    cudnn_header_path.c_str());
            return;
        }

        size_t cudnn_major = 0, cudnn_minor = 0, cudnn_patch = 0;
        sscanf(str_cudnn_major.c_str(), "#define CUDNN_MAJOR %zu",
               &cudnn_major);
        sscanf(str_cudnn_minor.c_str(), "#define CUDNN_MINOR %zu",
               &cudnn_minor);
        sscanf(str_cudnn_patch.c_str(), "#define CUDNN_PATCHLEVEL %zu",
               &cudnn_patch);

#ifdef WIN32
        std::string cudnn_lib_name =
                "cudnn64_" + std::to_string(cudnn_major) + ".dll";
#else
        std::string cudnn_lib_name =
                "libcudnn.so." + std::to_string(cudnn_major) + "." +
                std::to_string(cudnn_minor) + "." + std::to_string(cudnn_patch);
#endif

        auto cudnn_lib_path = find_file_in_envs_with_intmd(
                {"CUDNN_ROOT_DIR", "CUDNN_LIBRARY", "CUDA_PATH", ENV_PATH},
                cudnn_lib_name, {"lib64", "lib/x64"});
        if (cudnn_lib_path.empty()) {
            mgb_log_warn(
                    "%s not found. Please make sure cudnn install at "
                    "${CUDNN_LIBRARY}",
                    cudnn_lib_name.c_str());
        }
    }
}
}  // namespace

std::vector<std::string> _config::get_cuda_include_path() {
#if MGB_CUDA
    auto nvcc_path = get_nvcc_root_path();
    auto cudart_header_path =  nvcc_path + "../include/cuda_runtime.h";
    //! double check path_to_nvcc/../include/cuda_runtime.h exists
    auto ret = check_file_exist(cudart_header_path.c_str(), F_OK);
    if (ret) {
        mgb_throw(MegBrainError,
                  "%s not found. Please make sure your cuda toolkit install "
                  "right",
                  cudart_header_path.c_str());
    } else {
        return {nvcc_path + "..", nvcc_path + "../include"};
    }
#else
    mgb_throw(MegBrainError, "cuda disabled at compile time");
#endif
}

std::vector<std::string> _config::get_cuda_lib_path() {
#if MGB_CUDA
    auto nvcc_path = get_nvcc_root_path();
#ifdef WIN32
    auto cuda_version = get_local_cuda_version();
    auto cuda_major = cuda_version / 1000;
    auto cuda_minor = cuda_version % 10;
    auto cudart_lib_path = nvcc_path + "cudart64_" +
                           std::to_string(cuda_major * 10 + cuda_minor) +
                           ".dll";
#else
    auto cudart_lib_path = nvcc_path + "../lib64/libcudart.so";
#endif
    //! double check cudart_lib_path exists
    auto ret = check_file_exist(cudart_lib_path.c_str(), F_OK);
    if (ret) {
        mgb_throw(MegBrainError,
                  "%s not found. Please make sure your cuda toolkit install "
                  "right",
                  cudart_lib_path.c_str());
    } else {
#ifdef WIN32
        //! cudart64_101.dll locates at cuda/bin
        return {nvcc_path + "../lib/x64", nvcc_path};
#else
        return {nvcc_path + "../lib64"};
#endif
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

bool _config::is_local_cuda_env_ok() {
    check_cudnn_existence();
    if (get_nvcc_root_path().empty()) {
        return false;
    }
    return true;
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

std::vector<std::pair<size_t, std::string>> _config::dump_registered_oprs() {
#if MGB_ENABLE_DEBUG_UTIL
    return serialization::OprRegistry::dump_registries();
#else
    return {};
#endif
}

#if MGB_ENABLE_OPR_MM
/*! see definition : src/cpp/megbrain_config.h.
 * Create mm server. port 0 is permitted, leave zmqrpc to decide which port
 * should be used.
 */
int _config::create_mm_server(const std::string& server_addr, int port) {
    return create_zmqrpc_server(server_addr, port);
}

void _config::group_barrier(const std::string& server_addr,
        int port, uint32_t size, uint32_t rank) {
    mgb_assert(rank < size, "invalid rank %d", rank);
    auto group_mgr = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", server_addr.c_str(), port));
    uint32_t rsp = group_mgr->group_barrier(size, rank);
    mgb_assert(rsp != 0, "rank already registered: %d", rank);
    mgb_assert(size == rsp, "inconsistent size: %d, expect %d", size, rsp);
}

#else

int _config::create_mm_server(const std::string& server_addr, int port) {
    mgb_throw(mgb::MegBrainError, "OPR_MM suppport disable at compile time");
    return 0;
}

void _config::group_barrier(const std::string& server_addr,
        int port, uint32_t size, uint32_t rank) {
    mgb_throw(mgb::MegBrainError, "OPR_MM suppport disable at compile time");
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
