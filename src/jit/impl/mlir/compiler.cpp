/**
 * \file src/jit/impl/mlir/compiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "./compiler.h"
#include "./executable_cpu.h"
#include "./executable_cuda.h"
#include "./mlir_gen.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/jit/mlir/ir/dialect.h"
#include "megbrain/jit/mlir/ir/passes.h"

#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/GPU/Passes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Module.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/NVVMIR.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/TargetSelect.h>

using namespace mgb;
using namespace jit;

namespace {

struct LLVMInitializer {
    LLVMInitializer() {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
    }
};
static LLVMInitializer initializer;

#if MGB_CUDA
mlir::OwnedBlob compile_ptx_to_cubin(const std::string ptx, mlir::Location,
                                     llvm::StringRef) {
    OwnedBlob result = std::make_unique<std::vector<char>>(
            ptx.data(), ptx.data() + ptx.size());

    return result;
}

#endif

void add_cpu_lowering_pass(mlir::PassManager& manager) {
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(create_shape_inference_pass());
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
    }

    manager.addPass(create_lower_to_affine_pass());
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLoopFusionPass());
        opt_pm.addPass(mlir::createMemRefDataFlowOptPass());
    }
    manager.addPass(create_lower_to_llvm_pass());
}

#if MGB_CUDA
void add_cuda_lowering_pass(mlir::PassManager& manager, CompNode cn) {
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(create_shape_inference_pass());
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
    }
    manager.addPass(create_lower_to_gpu_pass());
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLoopFusionPass());
        opt_pm.addPass(mlir::createMemRefDataFlowOptPass());
    }
    manager.addPass(mlir::createGpuKernelOutliningPass());
    {
        auto& kernel_pm = manager.nest<gpu::GPUModuleOp>();
        kernel_pm.addPass(mlir::createLowerGpuOpsToNVVMOpsPass());

        auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
        kernel_pm.addPass(mlir::createConvertGPUKernelToBlobPass(
                mlir::translateModuleToNVVMIR, compile_ptx_to_cubin,
                "nvptx64-nvidia-cuda",
                ssprintf("sm_%d%d", prop.major, prop.minor), "+ptx60",
                MLIRCUDAExecutable::sm_blob_annotation));
    }
}
#endif

}  // namespace

/* ==================== MLIRCompiler ===================== */

thread_local mlir::MLIRContext MLIRCompiler::sm_ctx;

MLIRCompiler::MLIRCompiler(CompNode::DeviceType device_type)
        : m_device_type{device_type} {
    mlir::registerAllDialects();
    mlir::registerDialect<MgbDialect>();

#if MGB_CUDA
    if (m_device_type == CompNode::DeviceType::CUDA) {
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
    }
#endif
}

void MLIRCompiler::run_lowering_pass(mlir::OwningModuleRef& module,
                                     CompNode cn) {
    mgb_assert(cn.device_type() == m_device_type);
    mlir::PassManager manager(module->getContext());
    switch (m_device_type) {
        case CompNode::DeviceType::CPU:
            add_cpu_lowering_pass(manager);
            break;
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            add_cuda_lowering_pass(manager, cn);
            break;
#endif
        default:
            mgb_throw(InternalError, "Unsupport device type: %d",
                      static_cast<int>(m_device_type));
            break;
    }
    mgb_assert(mlir::succeeded(manager.run(*module)));
}

std::unique_ptr<Executable> MLIRCompiler::do_compile(
        const InternalGraph& graph, const JITExecutor::Args& args) {
    MGB_MARK_USED_VAR(graph);
    MGB_MARK_USED_VAR(args);

    mlir::MLIRContext ctx;
    ctx.printStackTraceOnDiagnostic(true);
    ctx.printOpOnDiagnostic(true);

    auto&& res = mlir_gen(ctx, graph, args);
    mgb_assert(res.second, "failed to generate module");

    CompNode cn = args.owner->comp_node();
    run_lowering_pass(res.second, cn);
    switch (cn.device_type()) {
        case CompNode::DeviceType::CPU:
            return std::make_unique<MLIRCPUExecutable>(res.second,
                                                       res.first.str());
#if MGB_CUDA
        case CompNode::DeviceType::CUDA:
            return std::make_unique<MLIRCUDAExecutable>(res.second,
                                                        res.first.str());
#endif
        default:
            mgb_throw(InternalError, "Unsupport device type: %d",
                      static_cast<int>(cn.device_type()));
            return nullptr;
    }
}

size_t MLIRCompiler::get_nr_workspace_outputs(JITExecutor* opr) const {
    MGB_MARK_USED_VAR(opr);
    return 0;
}

void MLIRCompiler::init_workspace_size_infer(JITExecutor* opr) {
    MGB_MARK_USED_VAR(opr);
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
