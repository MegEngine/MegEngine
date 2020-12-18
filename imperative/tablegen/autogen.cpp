#include <iostream>
#include <unordered_map>
#include <functional>

#include "./helper.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

enum ActionType {
    None,
    CppHeader,
    CppBody,
    Pybind
};

// NOLINTNEXTLINE
llvm::cl::opt<ActionType> action(
    llvm::cl::desc("Action to perform:"),
    llvm::cl::values(clEnumValN(CppHeader, "gen-cpp-header",
                                "Generate operator cpp header"),
                     clEnumValN(CppBody, "gen-cpp-body",
                                "Generate operator cpp body"),
                     clEnumValN(Pybind, "gen-python-binding",
                                "Generate pybind11 python bindings")));

using MgbAttrWrapper = mlir::tblgen::MgbAttrWrapperBase;
using MgbEnumAttr = mlir::tblgen::MgbEnumAttrMixin;
using MgbHashableAttr = mlir::tblgen::MgbHashableAttrMixin;
using MgbAliasAttr = mlir::tblgen::MgbAliasAttrMixin;
using MgbOp = mlir::tblgen::MgbOpBase;
using MgbHashableOp = mlir::tblgen::MgbHashableOpMixin;

llvm::StringRef attr_to_ctype(const mlir::tblgen::Attribute& attr_) {
    // Note: we have already registered the corresponding attr wrappers
    // for following basic ctypes so we needn't handle them here
    /* auto&& attr_type_name = attr.getAttrDefName();
    if (attr_type_name == "UI32Attr") {
        return "uint32_t";
    }
    if (attr_type_name == "UI64Attr") {
        return "uint64_t";
    }
    if (attr_type_name == "I32Attr") {
        return "int32_t";
    }
    if (attr_type_name == "F32Attr") {
        return "float";
    }
    if (attr_type_name == "F64Attr") {
        return "double";
    }
    if (attr_type_name == "StrAttr") {
        return "std::string";
    }
    if (attr_type_name == "BoolAttr") {
        return "bool";
    }*/

    auto&& attr = llvm::cast<MgbAttrWrapper>(attr_);
    if (auto e = llvm::dyn_cast<MgbEnumAttr>(&attr)) {
        return e->getEnumName();
    }
    return attr.getUnderlyingType();
}

static void gen_op_def_c_header_single(raw_ostream &os, MgbOp& op) {
    os << formatv(
        "class {0} : public OpDefImplBase<{0}> {{\n"
        "    MGB_DYN_TYPE_OBJ_FINAL_DECL;\n\n"
        "public:\n",
        op.getCppClassName()
    );
    // handle enum alias
    for (auto &&i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            os << formatv(
                "    using {0} = {1};\n",
                attr->getEnumName(), attr->getUnderlyingType()
            );
        }
    }
    for (auto &&i : op.getMgbAttributes()) {
        auto defaultValue = i.attr.getDefaultValue().str();
        if (!defaultValue.empty()) {
            defaultValue = formatv(" = {0}", defaultValue);
        }
        os << formatv(
            "    {0} {1}{2};\n",
            attr_to_ctype(i.attr), i.name, defaultValue
        );
    }

    auto gen_ctor = [&](auto&& paramList, auto&& memInitList, auto&& body) {
        os << formatv(
            "    {0}({1}){2}{3}\n",
            op.getCppClassName(), paramList, memInitList, body
        );
    };

    gen_ctor("", "", " = default;");

    if (!op.getMgbAttributes().empty()) {
        std::vector<std::string> paramList, initList;
        for (auto &&i : op.getMgbAttributes()) {
            paramList.push_back(formatv(
                "{0} {1}_", attr_to_ctype(i.attr), i.name
            ));
            initList.push_back(formatv(
                "{0}({0}_)", i.name
            ));
        }
        gen_ctor(llvm::join(paramList, ", "),
                 ": " + llvm::join(initList, ", "),
                 " {}");
    }

    auto packedParams = op.getPackedParams();
    if (!packedParams.empty()) {
        std::vector<std::string> paramList, initList;
        for (auto &&p : packedParams) {
            auto&& paramFields = p.getFields();
            auto&& paramType = p.getFullName();
            auto&& paramName = formatv("packed_param_{0}", paramList.size());
            paramList.push_back(
                paramFields.empty() ? paramType.str()
                    : formatv("{0} {1}", paramType, paramName)
            );
            for (auto&& i : paramFields) {
                initList.push_back(formatv(
                    "{0}({1}.{0})", i.name, paramName
                ));
            }
        }
        for (auto&& i : op.getExtraArguments()) {
            paramList.push_back(formatv(
                "{0} {1}_", attr_to_ctype(i.attr), i.name
            ));
            initList.push_back(formatv(
                "{0}({0}_)", i.name
            ));
        }
        gen_ctor(llvm::join(paramList, ", "),
                 initList.empty() ? "" : ": " + llvm::join(initList, ", "),
                 " {}");
    }

    if (!packedParams.empty()) {
        for (auto&& p : packedParams) {
            auto accessor = p.getAccessor();
            if (!accessor.empty()) {
                os << formatv(
                    "    {0} {1}() const {{\n",
                    p.getFullName(), accessor
                );
                std::vector<llvm::StringRef> fields;
                for (auto&& i : p.getFields()) {
                    fields.push_back(i.name);
                }
                os << formatv(
                    "        return {{{0}};\n",
                    llvm::join(fields, ", ")
                );
                os << "    }\n";
            }
        }
    }

    if (auto decl = op.getExtraOpdefDecl()) {
        os << decl.getValue();
    }

    os << formatv(
        "};\n\n"
    );
}

static void gen_op_def_c_body_single(raw_ostream &os, MgbOp& op) {
    auto&& className = op.getCppClassName();
    os << formatv(
        "MGB_DYN_TYPE_OBJ_FINAL_IMPL({0});\n\n", className
    );
    auto formatMethImpl = [&](auto&& meth) {
        return formatv(
            "{0}_{1}_impl", className, meth
        );
    };
    std::vector<std::string> methods;
    if (auto hashable = llvm::dyn_cast<MgbHashableOp>(&op)) {
        os << "namespace {\n";

        // generate hash()
        mlir::tblgen::FmtContext ctx;
        os << formatv(
            "size_t {0}(const OpDef& def_) {{\n",
            formatMethImpl("hash")
        );
        os << formatv(
            "    auto op_ = def_.cast_final_safe<{0}>();\n"
            "    static_cast<void>(op_);\n",
            className
        );
        ctx.withSelf("op_");
        os << mlir::tblgen::tgfmt(hashable->getHashFunctionTemplate(), &ctx);
        os << "}\n";

        // generate is_same_st()
        os << formatv(
            "bool {0}(const OpDef& lhs_, const OpDef& rhs_) {{\n",
            formatMethImpl("is_same_st")
        );
        os << formatv(
            "    auto a_ = lhs_.cast_final_safe<{0}>(),\n"
            "         b_ = rhs_.cast_final_safe<{0}>();\n"
            "    static_cast<void>(a_);\n"
            "    static_cast<void>(b_);\n",
            className
        );
        os << mlir::tblgen::tgfmt(hashable->getCmpFunctionTemplate(), &ctx, "a_", "b_");
        os << "}\n";

        os << "} // anonymous namespace\n";

        methods.push_back("hash");
        methods.push_back("is_same_st");
    }
    if (!methods.empty()) {
        os << formatv(
            "OP_TRAIT_REG({0}, {0})", op.getCppClassName()
        );
        for (auto&& i : methods) {
            os << formatv(
                "\n    .{0}({1})", i, formatMethImpl(i)
            );
        }
        os << ";\n\n";
    }
}

struct PybindContext {
    std::unordered_map<unsigned int, std::string> enumAlias;
};

static void gen_op_def_pybind11_single(raw_ostream &os, MgbOp& op, PybindContext& ctx) {
    auto class_name = op.getCppClassName();
    os << formatv(
        "py::class_<{0}, std::shared_ptr<{0}>, OpDef> {0}Inst(m, \"{0}\");\n\n",
        class_name
    );
    for (auto&& i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            unsigned int enumID;
            if (auto alias = llvm::dyn_cast<MgbAliasAttr>(attr)) {
                auto&& aliasBase = alias->getAliasBase();
                enumID =
                    llvm::cast<MgbEnumAttr>(aliasBase)
                            .getBaseRecord()->getID();
            } else {
                enumID = attr->getBaseRecord()->getID();
            }
            auto&& enumAlias = ctx.enumAlias;
            auto&& iter = enumAlias.find(enumID);
            if (iter == enumAlias.end()) {
                os << formatv(
                    "py::enum_<{0}::{1}>({0}Inst, \"{1}\")",
                    class_name, attr->getEnumName()
                );
                std::vector<std::string> body;
                for (auto&& i: attr->getEnumMembers()) {
                    os << formatv(
                        "\n    .value(\"{2}\", {0}::{1}::{2})",
                        class_name, attr->getEnumName(), i
                    );
                    body.push_back(formatv(
                        "if (str == \"{2}\") return {0}::{1}::{2};",
                        class_name, attr->getEnumName(), i
                    ));
                }
                os << formatv(
                    "\n    .def(py::init([](const std::string& in) {"
                    "\n        auto&& str = normalize_enum(in);"
                    "\n        {0}"
                    "\n        throw py::cast_error(\"invalid enum value \" + in);"
                    "\n    }));\n",
                    llvm::join(body, "\n        ")
                );
                os << formatv(
                    "py::implicitly_convertible<std::string, {0}::{1}>();\n\n",
                    class_name, attr->getEnumName()
                );
                enumAlias.emplace(enumID, formatv(
                    "{0}Inst.attr(\"{1}\")", class_name, attr->getEnumName()
                ));
            } else {
                os << formatv(
                    "{0}Inst.attr(\"{1}\") = {2};\n\n",
                    class_name, attr->getEnumName(), iter->second
                );
            }
        }
    }
    // generate op class binding
    os << formatv("{0}Inst", class_name);
    bool hasDefaultCtor = op.getMgbAttributes().empty();
    if (!hasDefaultCtor) {
        os << "\n    .def(py::init<";
        std::vector<llvm::StringRef> targs;
        for (auto &&i : op.getMgbAttributes()) {
            targs.push_back(i.attr.getReturnType());
        }
        os << llvm::join(targs, ", ");
        os << ">()";
        for (auto &&i : op.getMgbAttributes()) {
            os << formatv(", py::arg(\"{0}\")", i.name);
            auto defaultValue = i.attr.getDefaultValue();
            if (!defaultValue.empty()) {
                os << formatv(" = {0}", defaultValue);
            } else {
                hasDefaultCtor = true;
            }
        }
        os << ")";
    }
    if (hasDefaultCtor) {
        os << "\n    .def(py::init<>())";
    }
    for (auto &&i : op.getMgbAttributes()) {
        os << formatv(
            "\n    .def_readwrite(\"{0}\", &{1}::{0})",
            i.name, class_name
        );
    }
    os << ";\n\n";
}

static void for_each_operator(raw_ostream &os, RecordKeeper &keeper,
        std::function<void(raw_ostream&, MgbOp&)> callback) {
    auto op_base_class = keeper.getClass("Op");
    ASSERT(op_base_class, "could not find base class Op");
    for (auto&& i: keeper.getDefs()) {
        auto&& r = i.second;
        if (r->isSubClassOf(op_base_class)) {
            auto op = mlir::tblgen::Operator(r.get());
            if (op.getDialectName().str() == "mgb") {
                std::cerr << "\033[34;15m" << "Generating " << r->getName().str() << "\033[0m" << std::endl;
                callback(os, llvm::cast<MgbOp>(op));
            }
        }
    }
}

static bool gen_op_def_c_header(raw_ostream &os, RecordKeeper &keeper) {
    for_each_operator(os, keeper, gen_op_def_c_header_single);
    return false;
}

static bool gen_op_def_c_body(raw_ostream &os, RecordKeeper &keeper) {
    for_each_operator(os, keeper, gen_op_def_c_body_single);
    return false;
}

static bool gen_op_def_pybind11(raw_ostream &os, RecordKeeper &keeper) {
    PybindContext ctx;
    using namespace std::placeholders;
    for_each_operator(os, keeper,
        std::bind(gen_op_def_pybind11_single, _1, _2, std::ref(ctx)));
    return false;
}

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
    return -1;
}