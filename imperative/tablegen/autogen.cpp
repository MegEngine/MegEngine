#include "./targets/cpp_class.h"
#include "./targets/macros.h"
#include "./targets/pybind11.h"
#include "./targets/python_c_extension.h"

namespace {

using namespace mlir::tblgen;

enum ActionType { None, CppHeader, CppBody, Pybind, CPython, EnumListMacro };

// NOLINTNEXTLINE
llvm::cl::opt<ActionType> action(
        llvm::cl::desc("Action to perform:"),
        llvm::cl::values(
                clEnumValN(CppHeader, "gen-cpp-header", "Generate operator cpp header"),
                clEnumValN(CppBody, "gen-cpp-body", "Generate operator cpp body"),
                clEnumValN(
                        Pybind, "gen-python-binding",
                        "Generate pybind11 python bindings"),
                clEnumValN(
                        CPython, "gen-python-c-extension",
                        "Generate python c extensions"),
                clEnumValN(
                        EnumListMacro, "gen-enum-list-macro",
                        "Generate enum param list macro")));

template <llvm::TableGenMainFn* MainFn>
llvm::TableGenMainFn* WrapMain() {
    return [](llvm::raw_ostream& os, llvm::RecordKeeper& records) -> bool {
        os << "// clang-format off\n";
        auto ret = MainFn(os, records);
        os << "// clang-format on\n";
        return ret;
    };
}

}  // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv);
    if (action == ActionType::CppHeader) {
        return TableGenMain(argv[0], WrapMain<&gen_op_def_c_header>());
    }
    if (action == ActionType::CppBody) {
        return TableGenMain(argv[0], WrapMain<&gen_op_def_c_body>());
    }
    if (action == ActionType::Pybind) {
        return TableGenMain(argv[0], WrapMain<&gen_op_def_pybind11>());
    }
    if (action == ActionType::CPython) {
        return TableGenMain(argv[0], WrapMain<&gen_op_def_python_c_extension>());
    }
    if (action == ActionType::EnumListMacro) {
        return TableGenMain(argv[0], WrapMain<&gen_enum_param_list_macro>());
    }
    return -1;
}
