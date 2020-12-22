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
#include "megbrain/utils/timer.h"

#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
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
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Pass.h>

#include <dlfcn.h>
#include <dirent.h>

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

std::unique_ptr<llvm::Module> translate_module_to_nvvm_ir_and_link_device(
        Operation* m, llvm::LLVMContext& llvmContext, llvm::StringRef name) {
    std::unique_ptr<llvm::Module> module = mlir::translateModuleToNVVMIR(m, llvmContext);
    auto get_device_path = []() -> std::string {
        auto cuda_path = getenv("CUDA_BIN_PATH");
        std::string device_dir;
        if (!cuda_path) {
            char cuda_lib_path[PATH_MAX];
            auto handle = dlopen("libcudart.so", RTLD_GLOBAL | RTLD_LAZY);
            mgb_assert(handle != nullptr, "%s", dlerror());
            mgb_assert(dlinfo(handle, RTLD_DI_ORIGIN, &cuda_lib_path) != -1,
                       "%s", dlerror());
            device_dir =
                    std::string(cuda_lib_path) + "/../../../nvvm/libdevice/";
            mgb_assert(!dlclose(handle), "fail to dlclose handle");
        } else {
            device_dir = std::string(cuda_path) + "/nvvm/libdevice/";
        }

        DIR* dirp;
        struct dirent* directory;
        dirp = opendir(device_dir.c_str());
        if (dirp) {
            while ((directory = readdir(dirp)) != nullptr) {
                if (!strncmp(directory->d_name, "libdevice", 9)) {
                    closedir(dirp);
                    return device_dir + std::string(directory->d_name);
                }
            }
            closedir(dirp);
        }
        return {};
    };

    //! load libdevice.bc
    llvm::SMDiagnostic err;
    auto libdevice_path = get_device_path();
    std::unique_ptr<llvm::Module> mlib = llvm::parseIRFile(
            libdevice_path.c_str(), err, module->getContext());
    if (mlib.get()) {
        mlib->setTargetTriple(module->getTargetTriple());
        mlib->setDataLayout(module->getDataLayout());

        RealTimer timer;
        mgb_assert(
                !llvm::Linker::linkModules(*module, std::move(mlib),
                                           llvm::Linker::Flags::LinkOnlyNeeded),
                "failed to parse ir file libdevice.bc");
        mgb_log("MLIR JIT: link libdevice.bc, used: %.3fms", timer.get_msecs());
    } else {
        mgb_log_warn("Fail to load bitcode file %s", libdevice_path.c_str());
    }
    return module;
}

#endif

void add_cpu_lowering_pass(mlir::PassManager& manager) {
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
    }
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(create_lower_to_affine_pass());
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLoopFusionPass());
        opt_pm.addPass(mlir::createMemRefDataFlowOptPass());
    }
    manager.addPass(create_lower_to_llvm_pass());
}

#if MGB_CUDA
void add_cuda_lowering_pass(mlir::PassManager& manager,
                            const std::string& target_chip) {
    {
        mlir::OpPassManager& opt_pm = manager.nest<mlir::FuncOp>();
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLoopFusionPass());
        opt_pm.addPass(mlir::createMemRefDataFlowOptPass());
    }
    manager.addPass(create_lower_to_gpu_pass());
    {
        mlir::OpPassManager& opt_pm = manager.nest<gpu::GPUModuleOp>();
        opt_pm.addPass(mlir::createLowerToCFGPass());
        opt_pm.addPass(mlir::createCanonicalizerPass());
        opt_pm.addPass(mlir::createCSEPass());
        opt_pm.addPass(mlir::createLowerGpuOpsToNVVMOpsPass());
        opt_pm.addPass(mlir::createConvertGPUKernelToBlobPass(
                translate_module_to_nvvm_ir_and_link_device,
                compile_ptx_to_cubin, "nvptx64-nvidia-cuda", target_chip,
                "+ptx60", MLIRCUDAExecutable::sm_blob_annotation));
    }
}
#endif

}  // namespace

/* ==================== MLIRCompiler ===================== */

thread_local mlir::MLIRContext MLIRCompiler::sm_ctx;

MLIRCompiler::MLIRCompiler(CompNode::DeviceType device_type)
        : m_device_type{device_type} {
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
    std::string target_chip;
    switch (m_device_type) {
        case CompNode::DeviceType::CPU:
            add_cpu_lowering_pass(manager);
            break;
#if MGB_CUDA
        case CompNode::DeviceType::CUDA: {
            auto&& prop =
                    CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
            std::string target_chip =
                    ssprintf("sm_%d%d", prop.major, prop.minor);
            add_cuda_lowering_pass(manager, target_chip);
            break;
        }
#endif
        default:
            mgb_throw(InternalError, "Unsupport device type: %d",
                      static_cast<int>(m_device_type));
            break;
    }
    RealTimer timer;
    mgb_assert(mlir::succeeded(manager.run(*module)));
    mgb_log("MLIR JIT: run lowering pass used: %.3f ms", timer.get_msecs());
}

std::unique_ptr<Executable> MLIRCompiler::do_compile(
        const InternalGraph& graph, const JITExecutor::Args& args) {
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<MgbDialect>();
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
