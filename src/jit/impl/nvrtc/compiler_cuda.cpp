/**
 * \file src/jit/impl/nvrtc/compiler_cuda.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./compiler_cuda.h"
#include "./codegen_cuda.h"

#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/jit/param_elem_visitor.h"
#include "megbrain/utils/persistent_cache.h"
#include "megbrain/utils/timer.h"

#if MGB_JIT && MGB_CUDA

#include <dlfcn.h>
#include <nvrtc.h>

using namespace mgb;
using namespace jit;

namespace {
std::string NVRTCCompile(const std::string& code, int cap_major,
                         int cap_minor) {
    auto get_cuda_include_opts = []() {
        auto cuda_path = getenv("CUDA_BIN_PATH");
        if (cuda_path) {
            std::string path1 = std::string("-I") + cuda_path + "/include";
            std::string path2 = std::string("-I") + cuda_path + "/../include";
            return std::vector<std::string>{path1, path2};
        } else {
            char cuda_lib_path[PATH_MAX];
            auto handle = dlopen("libcudart.so",
                                 RTLD_GLOBAL | RTLD_LAZY | RTLD_NOLOAD);
            mgb_assert(handle != nullptr, "%s", dlerror());
            mgb_assert(dlinfo(handle, RTLD_DI_ORIGIN, &cuda_lib_path) != -1,
                       "%s", dlerror());
            return std::vector<std::string>{std::string("-I") + cuda_lib_path +
                                            "/../include"};
        }
    };
    static std::vector<std::string> cuda_include_opts = get_cuda_include_opts();

    auto arch_opt =
            ssprintf("--gpu-architecture=compute_%d%d", cap_major, cap_minor);
    std::vector<const char*> opts;
    opts.push_back(arch_opt.c_str());
    for (auto& inc_path : cuda_include_opts)
        opts.push_back(inc_path.c_str());
    nvrtcProgram prog;
    MGB_NVRTC_CHECK(nvrtcCreateProgram(&prog, code.c_str(), nullptr, 0, nullptr,
                                       nullptr));
    std::unique_ptr<nvrtcProgram, void (*)(nvrtcProgram*)> prog_release{
            &prog,
            [](nvrtcProgram* p) { MGB_NVRTC_CHECK(nvrtcDestroyProgram(p)); }};
    nvrtcResult compile_res =
            nvrtcCompileProgram(prog, opts.size(), opts.data());
    size_t log_size;
    MGB_NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    std::string log;
    log.resize(log_size);
    MGB_NVRTC_CHECK(nvrtcGetProgramLog(prog, &log[0]));
    mgb_throw_if(compile_res != NVRTC_SUCCESS, SystemError,
                 "nvrtc compile error: %s\n========= source code\n%s",
                 log.c_str(), code.c_str());
    size_t ptx_size;
    MGB_NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx;
    ptx.resize(ptx_size);
    MGB_NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
    return ptx;
}

void make_fastdiv(Uint32Fastdiv& fdiv, uint32_t d) {
    mgb_assert(d);
    fdiv.m_divisor = d;
    constexpr uint32_t MAX_U32 = ~0u;
    fdiv.m_inc_dividend = 0;
    fdiv.m_divisor_is_not_1 = ~0u;
    if (!(d & (d - 1))) {
        // power of 2
        fdiv.m_mul = 1u << 31;
        int p = 0;
        while ((1u << p) < d)
            ++p;
        mgb_assert((1u << p) == d);
        fdiv.m_shift = p ? p - 1 : 0;
        if (d == 1)
            fdiv.m_divisor_is_not_1 = 0;
        return;
    }
    auto n_bound = uint64_t(d / 2 + 1) * MAX_U32;
    uint32_t shift = 32;
    while ((1ull << shift) < n_bound)
        ++shift;
    uint64_t mdst = 1ull << shift;
    int64_t delta = d - mdst % d;
    fdiv.m_mul = mdst / d + 1;
    if ((uint64_t)delta > d / 2) {
        delta -= d;
        --fdiv.m_mul;
        fdiv.m_inc_dividend = 1;
    }
    mgb_assert((uint64_t)fdiv.m_mul * d == mdst + delta);
    delta = delta >= 0 ? delta : -delta;
    mgb_assert((uint64_t)delta * MAX_U32 < mdst);
    fdiv.m_shift = shift - 32;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
template <int ndim>
void host_init_pvisitor(ParamElemVisitor<ndim>& pvis,
                        const TensorLayout& layout) {
    mgb_assert(layout.ndim && layout.ndim <= ndim);
    for (uint32_t i = 0; i < layout.ndim; ++i) {
        pvis.m_stride[i] = layout.stride[i];
        if (i + 1 < layout.ndim) {
            make_fastdiv(pvis.m_shape_highdim[i], layout.shape[i + 1]);
        }
    }
    for (int i = layout.ndim - 1; i < ndim - 1; ++i) {
        make_fastdiv(pvis.m_shape_highdim[i], 1);
    }
    for (int i = layout.ndim; i < ndim; ++i) {
        pvis.m_stride[i] = 0;
    }
}
#pragma GCC diagnostic pop

template <size_t out_dim>
void setup_and_launch(const JITExecutor* fusion_opr, CUfunction func,
                      int block_size) {
    auto&& args = fusion_opr->args();

    size_t nr_inps = args.inputs.size();
    bool copy_param_to_dev = nr_inps > CudaCompiler::MAX_CUDA_NR_INPUT;
    SmallVector<CUdeviceptr> datum(nr_inps + 1);

    SmallVector<ParamElemVisitor<out_dim>> pvisitors;
    pvisitors.reserve(nr_inps);

    for (size_t i = 0; i < args.inputs.size(); i++) {
        datum[i] = reinterpret_cast<CUdeviceptr>(
                args.inputs[i].from->dev_tensor().raw_ptr());
        host_init_pvisitor<out_dim>(pvisitors[i], args.inputs[i].layout);
    }
    datum[nr_inps] = reinterpret_cast<CUdeviceptr>(
            args.outputs[0].from->dev_tensor().as_megdnn().raw_ptr);
    size_t num_elements = args.outputs[0].layout.total_nr_elems();
    mgb_assert(num_elements <= UINT32_MAX,
               "Currently JIT only supports 32 bit of elememt size for better "
               "performance");
    int num_block = (num_elements - 1) / (block_size * 3) + 1;

    void* exec_args[3];
    exec_args[1] = &num_elements;

    void* datum_dev = nullptr;
    void* p_visitors_dev = nullptr;
    const CompNodeEnv& env =
            CompNodeEnv::from_comp_node(fusion_opr->comp_node());

    if (!copy_param_to_dev) {
        exec_args[0] = datum.data();
        exec_args[2] = pvisitors.data();
    } else {
        datum_dev = args.outputs[1].from->dev_tensor().as_megdnn().raw_ptr;
        MGB_CUDA_CHECK(cudaMemcpyAsync(
                datum_dev, datum.data(), (nr_inps + 1) * sizeof(CUdeviceptr),
                cudaMemcpyHostToDevice, env.cuda_env().stream));
        p_visitors_dev = args.outputs[2].from->dev_tensor().as_megdnn().raw_ptr;
        MGB_CUDA_CHECK(
                cudaMemcpyAsync(p_visitors_dev, pvisitors.data(),
                                nr_inps * sizeof(ParamElemVisitor<out_dim>),
                                cudaMemcpyHostToDevice, env.cuda_env().stream));
        exec_args[0] = &datum_dev;
        exec_args[2] = &p_visitors_dev;
    }

    MGB_CUDA_CU_CHECK(cuLaunchKernel(func, num_block, 1, 1, block_size, 1, 1, 0,
                                     env.cuda_env().stream, exec_args, 0));
}
}  // namespace

void mgb::jit::_on_nvrtc_error(const char* expr, nvrtcResult nvrtc_res,
                               const char* file, const char* func, int line) {
    mgb_throw(CudaError, "nvrtc error %d: %s (%s at %s:%s:%d)", int(nvrtc_res),
              nvrtcGetErrorString(nvrtc_res), expr, file, func, line);
}

/* =================== CudaExecutable ==================== */

CudaExecutable::CudaExecutable(std::string source, std::string name)
        : m_source{std::move(source)}, m_name{std::move(name)} {}

void CudaExecutable::execute(JITExecutor* fusion_opr) {
    FuncCache* func;
    auto cn = fusion_opr->comp_node();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    {
        MGB_LOCK_GUARD(m_mtx);
        func = &m_func_cache[{prop.major, prop.minor}];
    }
    {
        MGB_LOCK_GUARD(func->mtx);
        if (func->ptx.empty()) {
            func->compile(
                    "jit:nvrtc:" +
                            PersistentCache::make_category_from_comp_node(cn),
                    prop.major, prop.minor, this);
        }
    }
    func->exec(fusion_opr, this);
}

void CudaExecutable::FuncCache::compile(const std::string& cache_category,
                                        int major, int minor,
                                        const CudaExecutable* cuda_exe) {
    RealTimer timer;
    auto&& cache = PersistentCache::inst();
    PersistentCache::Blob key{cuda_exe->m_source.data(),
                              cuda_exe->m_source.size()};
    auto ptx_cache = cache.get(cache_category, key);
    if (ptx_cache.valid()) {
        ptx.assign(static_cast<const char*>(ptx_cache->ptr), ptx_cache->size);
    } else {
        ptx = NVRTCCompile(cuda_exe->m_source, major, minor);
        ptx_cache = PersistentCache::Blob{ptx.data(), ptx.size()};
        cache.put(cache_category, key, ptx_cache.val());
    }
    mgb_log("NVRTC JIT: compile %s for %d.%d: source_len=%zu ptx_len=%zu "
            "time=%.3fms",
            cuda_exe->m_name.c_str(), major, minor, key.size, ptx.size(),
            timer.get_msecs());
}

void CudaExecutable::FuncCache::exec(const JITExecutor* fusion_opr,
                                     const CudaExecutable* cuda_exe) {
    Func* func;
    {
        MGB_LOCK_GUARD(mtx);
        auto ins = cn2func.insert({fusion_opr->comp_node(), {}});
        func = &ins.first->second;
        if (ins.second) {
            MGB_CUDA_CU_CHECK(cuModuleLoadData(&func->module, ptx.data()));
            MGB_CUDA_CU_CHECK(cuModuleGetFunction(&func->func, func->module,
                                                  cuda_exe->m_name.c_str()));
            int min_grid_size = 0;
            MGB_CUDA_CU_CHECK(cuOccupancyMaxPotentialBlockSize(
                    &min_grid_size, &func->block_size, func->func, nullptr, 0,
                    0));
        }
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
    int out_dim = fusion_opr->args().outputs[0].layout.ndim;
#define cb_outdim(EXPECTED_OUTDIM)                                \
    if (EXPECTED_OUTDIM == out_dim) {                             \
        setup_and_launch<EXPECTED_OUTDIM>(fusion_opr, func->func, \
                                          func->block_size);      \
        return;                                                   \
    }
#pragma GCC diagnostic push
    cb_outdim(1);
    cb_outdim(2);
    cb_outdim(3);
    cb_outdim(4);
    mgb_throw(InternalError, "unsupported out_dim=%zu",
              static_cast<size_t>(out_dim));
#undef cb_outdim
}

CudaExecutable::~CudaExecutable() {
    for (auto&& i : m_func_cache) {
        for (auto&& j : i.second.cn2func) {
            j.first.activate();
            if (auto m = j.second.module) {
                cuModuleUnload(m);
            }
        }
    }
}

/* ==================== CudaCompiler ===================== */

std::unique_ptr<Executable> CudaCompiler::do_compile(
        const InternalGraph& graph, const JITExecutor::Args& args) {
    bool copy_param_to_dev = graph.placeholders().size() > MAX_CUDA_NR_INPUT;
    if (copy_param_to_dev) {
        mgb_log_warn(
                "Too many[%zu] inputs, which exceeds the limit[%zu].  JIT "
                "kernel function's parameters will be "
                "put in GPU global memory.",
                graph.placeholders().size(), MAX_CUDA_NR_INPUT);
    }
    std::string source, kernel_name;
    std::tie(kernel_name, source) =
            codegen_cuda(graph, args, copy_param_to_dev);
    auto ret = std::make_unique<CudaExecutable>(std::move(source),
                                                std::move(kernel_name));
    return ret;
}

size_t CudaCompiler::get_nr_workspace_outputs(JITExecutor* opr) const {
    if (opr->input().size() > MAX_CUDA_NR_INPUT) {
        return 2;
    }
    return 0;
}

void CudaCompiler::init_workspace_size_infer(JITExecutor* opr) {
    if (opr->output().size() == 3) {
        using namespace cg::static_infer;
        auto&& mgr = opr->owner_graph()->static_infer_manager();
        TensorShape output_shape1(
                {(opr->input().size() + 1) * sizeof(unsigned long long)});
        mgr.register_shape_infer(opr->output(1),
                                 ShapeInferDesc::make_const(output_shape1));
        TensorShape output_shape2(
                {opr->input().size() * sizeof(ParamElemVisitor<4>)});
        mgr.register_shape_infer(opr->output(2),
                                 ShapeInferDesc::make_const(output_shape2));
    }
}

#endif  // MGB_JIT && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
