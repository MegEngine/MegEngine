/**
 * \file tools/mlir/mgb-opt/mgb-opt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/jit/mlir/ir/dialect.h"
#include "megbrain/jit/mlir/ir/passes.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/AsmState.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>

using namespace llvm;
using namespace mlir;

//! TODO: Implement a custom MlirOptMain that supports the following flags.
static cl::opt<bool> print_mlir{
        "print-mlir",
        cl::desc("Prints MLIR IR after translation"),
        cl::init(false),
};

static cl::list<std::string> input_values{
        "input-value",
        cl::desc("Input shapes and optional values"),
        cl::ZeroOrMore,
};

static cl::opt<std::string> input_values_file{
        "input-value-file",
        cl::desc("Provides a file for input shapes and optional values (see "
                 "ParseToVariantListFromFile in vm_util.h for details)"),
        cl::init(""),
};

static cl::opt<bool> run{
        "run",
        cl::desc("Runs the module (vs. just compiling and verifing)"),
        cl::init(true),
};

static cl::list<std::string> run_args{
        "run-arg",
        cl::desc("Argument passed to the execution flag parser"),
        cl::ZeroOrMore,
};

namespace mgb {
namespace jit {
void register_test_mgb_to_affine_lowering_pass();
void register_test_affine_to_llvm_lowering_pass();
}  // namespace jit
}  // namespace mgb

int main(int argc, char** argv) {
    mlir::registerAllPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mgb::jit::MgbDialect>();

    mgb::jit::register_test_mgb_to_affine_lowering_pass();
    mgb::jit::register_test_affine_to_llvm_lowering_pass();

    return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver", registry));
}
