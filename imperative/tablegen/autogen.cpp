/**
 * \file imperative/tablegen/autogen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./targets/cpp_class.h"
#include "./targets/pybind11.h"
#include "./targets/python_c_extension.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

enum ActionType {
    None,
    CppHeader,
    CppBody,
    Pybind,
    CPython
};

// NOLINTNEXTLINE
llvm::cl::opt<ActionType> action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(clEnumValN(CppHeader, "gen-cpp-header",
                                "Generate operator cpp header"),
                     clEnumValN(CppBody, "gen-cpp-body",
                                "Generate operator cpp body"),
                     clEnumValN(Pybind, "gen-python-binding",
                                "Generate pybind11 python bindings"),
                     clEnumValN(CPython, "gen-python-c-extension",
                                "Generate python c extensions")));

using namespace mlir::tblgen;

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv);
    if (action == ActionType::CppHeader) {
        return TableGenMain(argv[0], &gen_op_def_c_header);
    }
    if (action == ActionType::CppBody) {
        return TableGenMain(argv[0], &gen_op_def_c_body);
    }
    if (action == ActionType::Pybind) {
        return TableGenMain(argv[0], &gen_op_def_pybind11);
    }
    if (action == ActionType::CPython) {
        return TableGenMain(argv[0], &gen_op_def_python_c_extension);
    }
    return -1;
}
