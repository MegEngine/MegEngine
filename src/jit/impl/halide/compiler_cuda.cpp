/**
 * \file src/jit/impl/halide/compiler_cuda.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./compiler_cuda.h"

#if MGB_JIT_HALIDE && MGB_CUDA

#include "../nvrtc/compiler_cuda.h"
#include "./ast_hl.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/jit/utils.h"
#include "megbrain/utils/timer.h"

#include <HalideRuntimeCuda.h>

using namespace mgb;
using namespace jit;
using namespace Halide;

/* =================== HalideCudaTargetTrait ==================== */

struct HalideCudaTargetTrait::UserData
        : public HalideExecutable::TargetTraitUserData {
    DeviceProp dev_prop;  //!< dev prop used to generate schedule the func
    Halide::Pipeline pipeline;
    std::mutex mtx;
};

HalideCudaTargetTrait::FeatureSet HalideCudaTargetTrait::features(
        CompNode comp_node) const {
    FeatureSet set;
    set.set(Target::CUDA);
    auto&& prop = CompNodeEnv::from_comp_node(comp_node).cuda_env().device_prop;
    auto in = [ver = prop.major * 10 + prop.minor](int low, int high) {
        return ver >= low && ver < high;
    };
    if (in(30, 32)) {
        set.set(Target::CUDACapability30);
    } else if (in(32, 35)) {
        set.set(Target::CUDACapability32);
    } else if (in(35, 40)) {
        set.set(Target::CUDACapability35);
    } else if (in(50, 61)) {
        set.set(Target::CUDACapability50);
    } else if (in(61, 70)) {
        set.set(Target::CUDACapability61);
    } else {
        mgb_log_warn("cuda capability(%d.%d) not support for Halide, using compute capability 6.1",
                  prop.major, prop.minor);
        set.set(Target::CUDACapability61);
    }
    return set;
}

HalideCudaTargetTrait::FunctionHandle HalideCudaTargetTrait::compile_and_load(
        CompNode comp_node, Halide::Target target,
        const HalideExecutable& hl_exec) {
    auto&& dev_prop = get_dev_prop(comp_node);
    auto func_name = next_kernel_name();
    auto&& helper = ExecutableHelper::get();
    auto make_ud =
            [&]() -> std::unique_ptr<HalideExecutable::TargetTraitUserData> {
        auto ret = std::make_unique<UserData>();
        ret->dev_prop = dev_prop;
        ret->pipeline =
                gen_halide_pipeline_schedule(hl_exec.halide_output(), dev_prop);
        return ret;
    };
    auto ud = static_cast<UserData*>(user_data(hl_exec, make_ud));
    // since halide func and schedule are coupled, we need to copy the func to
    // use a different schedule
    mgb_throw_if(dev_prop.max_threads_per_block !=
                         ud->dev_prop.max_threads_per_block,
                 InternalError,
                 "halide on multiple devices with different "
                 "max_threads_per_block is currently not supported");
    auto&& pipeline = ud->pipeline;

    auto halide_inputs = hl_exec.halide_inputs();
    RealTimer timer;
    {
        // this compile seems not thread safe
        MGB_LOCK_GUARD(ud->mtx);

        pipeline.compile_to_object(helper.realpath(func_name + ".o"),
                                   halide_inputs, func_name, target);
        if (ExecutableHelper::keep_interm()) {
            pipeline.compile_to_lowered_stmt(
                    helper.realpath(func_name + ".stmt"), halide_inputs, Text,
                    target);
        }
    }
    auto time_compile = timer.get_msecs_reset();

    FunctionHandle ret;
    ret.init_uctx_map();
    auto obj_name = func_name + ".o";
    ret.dl_handle = helper.link_and_load(
            {HalideCudaCompiler::cuda_runtime_lib(), obj_name},
            func_name + ".so");
    helper.remove_interm(obj_name);

    helper.resolve_func(ret.get_device_interface, ret.dl_handle,
                        "halide_cuda_device_interface");
    helper.resolve_func(ret.execute, ret.dl_handle, func_name + "_argv");
    helper.resolve_func(ret.device_release, ret.dl_handle,
                        "halide_cuda_device_release");
    auto time_link = timer.get_msecs_reset();
    mgb_log("Halide CUDA JIT: compile %s for %s: time_compile=%.3fms "
            "time_link=%.3fms",
            func_name.c_str(), comp_node.to_string().c_str(), time_compile,
            time_link);
    return ret;
}

void* HalideCudaTargetTrait::get_user_context(CompNode comp_node) {
    return &(get_dev_prop(comp_node).ctx);
}

HalideCudaTargetTrait::DeviceProp& HalideCudaTargetTrait::get_dev_prop(
        CompNode comp_node) {
    MGB_LOCK_GUARD(m_mtx);
    auto&& ret = m_cn2prop[comp_node];
    if (ret.max_threads_per_block == -1) {
        auto&& env = CompNodeEnv::from_comp_node(comp_node).cuda_env();
        comp_node.activate();
        MGB_CUDA_CU_CHECK(cuCtxGetCurrent(&(ret.ctx.ctx)));
        ret.ctx.strm = env.stream;
        ret.max_threads_per_block = env.device_prop.maxThreadsPerBlock;
    }
    return ret;
}

Halide::Pipeline HalideCudaTargetTrait::gen_halide_pipeline_schedule(
        const ast_hl::AstNodePtr& dst_output, const DeviceProp& device_prop) {
#if 1
    using namespace ast_hl;
    // traverse inline
    std::unordered_set<AstNodePtr> visited;
    std::queue<AstNodePtr> q;
    for (auto inp : dst_output->m_inputs) {
        q.push(inp);
    }

    std::unordered_set<ReduceOp*> reduce_set;
    while (!q.empty()) {
        auto top = q.front();
        if (visited.count(top)) {
            q.pop();
            continue;
        }
        for (auto inp : top->m_inputs) {
            q.push(inp);
        }
        if (!top->same_type<InputDevValueOp>() && !top->same_type<ReduceOp>() &&
            !top->same_type<InputHostValueShapeOp>() &&
            !top->same_type<BroadcastOp>()) {
            top->m_func.compute_inline();
        }
        if (auto reduce_opr = try_cast_as_op<ReduceOp>(top.get())) {
            reduce_set.insert(reduce_opr);
        }
        visited.insert(top);
        q.pop();
    }

    std::vector<Func> outputs;
    auto process_reduce = [&](Func f, Var tx) {
        for (auto&& reduce_opr : reduce_set) {
            if (reduce_opr->m_comp.defined()) {
                reduce_opr->m_comp.compute_at(f, tx);
            }
            reduce_opr->m_func.compute_at(f, tx);
        }
    };

    auto schedule_elemwise_like = [&process_reduce, &outputs,
                                   &device_prop](const AstNodePtr& output) {
        auto& f = output->m_func;
        auto vars = f.args();
        auto&& layout = output->m_layout;
        size_t total_nr_elems = layout.total_nr_elems();
        mgb_assert(vars.size() == layout.ndim);
        for (int i = layout.ndim - 1; i >= 0; i--) {
            f.bound(vars[layout.ndim - 1 - i], 0, static_cast<int>(layout[i]));
        }

        Var fused = vars[0];
        for (size_t i = 1; i < vars.size(); i++) {
            output->m_func.fuse(fused, vars[i], fused);
        }
        const int max_blocks = 65536;
        const int max_threads_num = device_prop.max_threads_per_block;
        bool need_block_split =
                total_nr_elems >
                static_cast<size_t>(max_blocks * max_threads_num);
        const int bt = max_blocks * max_threads_num;

        if (need_block_split) {
            Var xo, xi;
            Var bx, tx;
            f.split(fused, xo, xi, bt, TailStrategy::GuardWithIf);
            f.split(xi, bx, tx, Expr{max_threads_num},
                    TailStrategy::GuardWithIf);
            f.reorder(xo, tx, bx);
            f.unroll(xo);
            f.gpu_threads(tx);
            f.gpu_blocks(bx);
            process_reduce(f, tx);
        } else {
            Var bx, tx;
            f.split(fused, bx, tx, max_threads_num, TailStrategy::GuardWithIf);
            f.gpu_threads(tx);
            f.gpu_blocks(bx);
            process_reduce(f, tx);
        }
        outputs.push_back(f);
    };

    auto schedule_reduce = [&process_reduce, &outputs,
                            &device_prop](const AstNodePtr& output) {
        auto& f = output->m_func;
        auto& c = try_cast_as_op<ReduceOp>(output.get())->m_comp;
        auto vars = f.args();
        std::vector<Expr> exprs;
        Func real_out;
        for (auto var : vars) {
            exprs.emplace_back(var);
        }
        real_out(vars) = f(exprs);

        auto layout = output->m_layout;
        size_t total_nr_elems = layout.total_nr_elems();
        for (int i = layout.ndim - 1; i >= 0; i--) {
            real_out.bound(vars[layout.ndim - 1 - i], 0,
                           static_cast<int>(layout[i]));
        }

        Var fused = vars[0];
        for (size_t i = 1; i < vars.size(); i++) {
            real_out.fuse(fused, vars[i], fused);
        }
        const int max_blocks = 65536;
        const int max_threads_num = device_prop.max_threads_per_block;
        bool need_block_split =
                total_nr_elems >
                static_cast<size_t>(max_blocks * max_threads_num);
        const int bt = max_blocks * max_threads_num;

        if (need_block_split) {
            Var xo, xi;
            Var bx, tx;
            real_out.split(fused, xo, xi, bt, TailStrategy::GuardWithIf);
            real_out.split(xi, bx, tx, Expr{max_threads_num},
                           TailStrategy::GuardWithIf);
            real_out.reorder(xo, tx, bx);
            real_out.unroll(xo);
            real_out.gpu_threads(tx);
            real_out.gpu_blocks(bx);
            f.compute_at(real_out, tx);
            if (c.defined())
                c.compute_at(real_out, tx);
            process_reduce(real_out, tx);
        } else {
            Var bx, tx;
            real_out.split(fused, bx, tx, max_threads_num,
                           TailStrategy::GuardWithIf);
            real_out.gpu_threads(tx);
            real_out.gpu_blocks(bx);
            f.compute_at(real_out, tx);
            if (c.defined())
                c.compute_at(real_out, tx);
            process_reduce(real_out, tx);
        }
        outputs.push_back(real_out);
    };

    if (dst_output->same_type<ReduceOp>()) {
        schedule_reduce(dst_output);
    } else {
        schedule_elemwise_like(dst_output);
    }

    return Pipeline(outputs);
#else
    return Pipeline(dst_output->m_func);
#endif
}

/* ==================== HalideCudaCompiler ===================== */

std::unique_ptr<Executable> HalideCudaCompiler::do_compile(
        const InternalGraph& graph, const JITExecutor::Args& args) {
    return std::make_unique<HalideExecutable>(m_trait, graph, args);
}

const std::string& HalideCudaCompiler::cuda_runtime_lib() {
    static const char* const source = R"(
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

namespace {
struct HalideUserContext {
    CUcontext ctx;
    CUstream strm;
};

HalideUserContext* check_user_context(void* user_context) {
    if (!user_context) {
        fprintf(stderr, "user_context not provided\n");
        abort();
    }
    return static_cast<HalideUserContext*>(user_context);
}
} // anonymous namespace

extern "C" int halide_cuda_acquire_context(void* user_context, CUcontext* ctx,
                                           bool create) {
    if (!user_context && !create) {
        // called from halide_cuda_cleanup()
        return 1;
    }
    *ctx = check_user_context(user_context)->ctx;
    return 0;
}

extern "C" int halide_cuda_release_context(void* user_context) {
    return 0;
}

extern "C" int halide_cuda_get_stream(void* user_context, CUcontext ctx,
                                      CUstream* stream) {
    *stream = check_user_context(user_context)->strm;
    return 0;
}
)";

    static std::string name =
            ExecutableHelper::get().compile_cpp_source_secondary(
                    source, "halide_cuda_runtime_override");
    return name;
}

#endif  // MGB_JIT_HALIDE && MGB_CUDA

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
