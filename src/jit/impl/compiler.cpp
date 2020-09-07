/**
 * \file src/jit/impl/compiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./halide/compiler_cuda.h"
#include "./nvrtc/compiler_cuda.h"
#include "./mlir/compiler.h"

#include "megbrain/jit/compiler.h"
#include "megbrain/utils/hash.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;

namespace {
class CompilerHolder final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

public:
    std::mutex mtx;
    ThinHashMap<CompNode::DeviceType, std::unique_ptr<Compiler>> dev2compiler;
};
MGB_TYPEINFO_OBJ_IMPL(CompilerHolder);

}  // anonymous namespace

class Compiler::EmptyCompiler final : public Compiler {
public:
    Property property() const {
        return {Property::Flag::NONE, JITFeatureBits::NONE, 100};
    }

    size_t get_nr_workspace_outputs(JITExecutor*) const { return 0; }

    void init_workspace_size_infer(JITExecutor*) {}

    std::unique_ptr<Executable> do_compile(const InternalGraph&,
                                           const JITExecutor::Args&) {
        mgb_throw(InternalError, "EmptyCompiler should not be used");
    }
};

bool Compiler::is_supported_device(CompNode::DeviceType device) {
    switch (device) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            return true;
#endif
        case CompNode::DeviceType::CPU:
            return true;
        default:
            return false;
    }
}

Compiler* Compiler::get(ComputingGraph& graph, CompNode comp_node) {
    static EmptyCompiler empty_compiler;
    if (comp_node == CompNode::default_cpu()) {
        // oprs in the internal graph are on default cpu; this case handles
        // nested JITExecutor
        return &empty_compiler;
    }

    CompilerHolder* holder;
    {
        static std::mutex mtx;
        MGB_LOCK_GUARD(mtx);
        holder = graph.options()
                         .user_data.get_user_data_or_create<CompilerHolder>();
    }
    MGB_LOCK_GUARD(holder->mtx);
    auto&& compiler = holder->dev2compiler[comp_node.device_type()];
    auto backend = MGB_GETENV("MGB_JIT_BACKEND");
    if (!compiler) {
        switch (comp_node.device_type()) {
#if MGB_CUDA
            case CompNode::DeviceType::CUDA:
#if MGB_JIT_HALIDE
                if (!backend || !strcmp(backend, "HALIDE")) {
                    compiler = std::make_unique<HalideCudaCompiler>();
                    break;
                }
#endif
#if MGB_JIT_MLIR
                if (!backend || !strcmp(backend, "MLIR")) {
                    compiler = std::make_unique<MLIRCompiler>(
                            CompNode::DeviceType::CUDA);
                    break;
                }
#endif
                if (!backend || !strcmp(backend, "NVRTC")) {
                    compiler = std::make_unique<CudaCompiler>();
                    break;
                }
                mgb_throw(InternalError, "No compiler support for cuda");
                break;
#endif
            case CompNode::DeviceType::CPU:
#if MGB_JIT_MLIR
                if (!backend || !strcmp(backend, "MLIR")) {
                    compiler = std::make_unique<MLIRCompiler>(
                            CompNode::DeviceType::CPU);
                    break;
                }
#endif
                mgb_throw(InternalError, "No compiler support for cpu");
                break;
            default:
                mgb_throw(InternalError,
                          "unsupported JIT config: "
                          "comp_node=%s backend_setting=%s",
                          comp_node.to_string().c_str(), backend);
        }
    }

    return compiler.get();
}

Executable* Compiler::compile(JITExecutor* opr) {
    MGB_LOCK_GUARD(m_mtx);
    auto&& args = opr->args();
    auto&& args_cache = m_expr_cache[&(opr->internal_graph())];
    auto q = args_cache.get(args);
    if (q.first) {
        *q.second = do_compile(opr->internal_graph(), opr->args());
    }
    return q.second->get();
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
