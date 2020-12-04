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
            "    auto&& op_ = def_.cast_final_safe<{0}>();\n"
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
            "    auto &&a_ = lhs_.cast_final_safe<{0}>(),\n"
            "         &&b_ = rhs_.cast_final_safe<{0}>();\n"
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

struct EnumContext {
    std::unordered_map<unsigned int, std::pair<llvm::StringRef, llvm::StringRef>> enumAlias;
};

static void gen_op_def_pybind11_single(raw_ostream &os, MgbOp& op, EnumContext& ctx) {
    auto className = op.getCppClassName();
    os << formatv(
        "py::class_<{0}, std::shared_ptr<{0}>, OpDef> {0}Inst(m, \"{0}\");\n\n",
        className
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
                    className, attr->getEnumName()
                );
                std::vector<std::string> body;
                for (auto&& i: attr->getEnumMembers()) {
                    os << formatv(
                        "\n    .value(\"{2}\", {0}::{1}::{2})",
                        className, attr->getEnumName(), i
                    );
                    body.push_back(formatv(
                        "if (str == \"{2}\") return {0}::{1}::{2};",
                        className, attr->getEnumName(), i
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
                    className, attr->getEnumName()
                );
                enumAlias.emplace(enumID,
                    std::make_pair(className, attr->getEnumName()));
            } else {
                os << formatv(
                    "{0}Inst.attr(\"{1}\") = {2}Inst.attr(\"{3}\");\n\n",
                    className, attr->getEnumName(),
                    iter->second.first, iter->second.second
                );
            }
        }
    }
    // generate op class binding
    os << formatv("{0}Inst", className);
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
            i.name, className
        );
    }
    os << ";\n\n";
}

static void gen_op_def_python_c_extension_single(raw_ostream &os, MgbOp& op, EnumContext& ctx) {
    auto className = op.getCppClassName();
    std::string body;

    // generate PyType for enum class member
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
            auto enumName = attr->getEnumName();
            body += "{\n";
            body += formatv(
                "auto& e_type = EnumWrapper<{0}::{1}>::type;", className, enumName
            );
            if (iter == enumAlias.end()) {
                os << formatv(
                    "template<> PyTypeObject EnumWrapper<{0}::{1}>::type={{};\n",
                    className, enumName);
                os << formatv(
                    "template<> const char* EnumWrapper<{0}::{1}>::name = \"{0}.{1}\";\n",
                    className, enumName);
                std::vector<std::string> pairStr;
                for (auto&& i: attr->getEnumMembers()) {
                    pairStr.push_back(formatv(
                        "{{normalize_enum(\"{2}\"), {0}::{1}::{2}}",
                        className, enumName, i));
                }
                os << formatv(R"(
template<> std::unordered_map<std::string, {0}::{1}>
EnumWrapper<{0}::{1}>::str2type = {{
    {2}
};
)", className, enumName, llvm::join(pairStr, ", "));
                pairStr.clear();
                for (auto&& i: attr->getEnumMembers()) {
                    pairStr.push_back(formatv(
                        "{{{0}::{1}::{2}, normalize_enum(\"{2}\")}",
                        className, enumName, i));
                }
                os << formatv(R"(
template<> std::unordered_map<{0}::{1}, std::string>
EnumWrapper<{0}::{1}>::type2str = {{
    {2}
};
)", className, enumName, llvm::join(pairStr, ", "));
                body += formatv(R"(
    e_type = {{PyVarObject_HEAD_INIT(NULL, 0)};
    e_type.tp_name = "megengine.core._imperative_rt.ops.{0}.{1}";
    e_type.tp_basicsize = sizeof(EnumWrapper<{0}::{1}>);
    e_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    e_type.tp_doc = "{0}.{1}";
    e_type.tp_base = &PyBaseObject_Type;
    e_type.tp_repr = EnumWrapper<{0}::{1}>::py_repr;
    e_type.tp_richcompare = EnumWrapper<{0}::{1}>::tp_richcompare;
    mgb_assert(PyType_Ready(&e_type) >= 0);
)", className, enumName);
                for (auto&& i: attr->getEnumMembers()) {
                    body += formatv(R"({{
    PyObject* inst = e_type.tp_alloc(&e_type, 0);
    reinterpret_cast<EnumWrapper<{0}::{1}>*>(inst)->value = {0}::{1}::{2};
    mgb_assert(PyDict_SetItemString(e_type.tp_dict, "{2}", inst) >= 0);
})", className, enumName, i);
                }
                enumAlias.emplace(enumID, std::make_pair(className, enumName));
            }
            body += formatv(R"(
    PyType_Modified(&e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "{0}", reinterpret_cast<PyObject*>(&e_type)) >= 0);
)", enumName);
            body += "}\n";
        }
    }

    // generate getsetters
    std::vector<std::string> getsetters;
    for (auto &&i : op.getMgbAttributes()) {
        getsetters.push_back(formatv(
            "{{\"{1}\", py_get_generic({0}, {1}), py_set_generic({0}, {1}), \"{1}\", NULL},",
            className, i.name));
    }

    // generate tp_init
    std::string initBody;
    if (!op.getMgbAttributes().empty()) {
        initBody += "static const char* kwlist[] = {";
        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            initBody += formatv("\"{0}\", ", attr.name);
        });
        initBody += "NULL};\n";
        initBody += "    PyObject ";
        std::vector<std::string> attrs;
        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            attrs.push_back(formatv("*{0} = NULL", attr.name));
        });
        initBody += llvm::join(attrs, ", ") + ";\n";
        initBody += "    if (!PyArg_ParseTupleAndKeywords(args, kwds, \"|";
        initBody += std::string(op.getMgbAttributes().size(), 'O');
        initBody += "\", const_cast<char**>(kwlist)";
        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            initBody += formatv(" ,&{0}", attr.name);
        });
        initBody += "))\n";
        initBody += "    return -1;\n";
        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            initBody += formatv(R"(
    if ({1}) {{
        try {{
            reinterpret_cast<PyOp({0})*>(self)->inst().{1} =
                pyobj_convert_generic<decltype({0}::{1})>::from({1});
        } catch(py::error_already_set& e) {{
            e.restore();
            return -1;
        } catch(py::builtin_exception& e) {{
            e.set_error();
            return -1;
        } catch(...) {{
            PyErr_SetString(PyExc_RuntimeError, "Unknown Error");
            return -1;
        }
    }
)", className, attr.name);
        });
    }
    initBody += "\n    return 0;";

    os << formatv(R"(
PyOpDefBegin({0}) // {{
    static PyGetSetDef py_getsetters[];
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
// };
PyOpDefEnd({0})
PyGetSetDef PyOp({0})::py_getsetters[] = {{
    {1}
    {{NULL}  /* Sentinel */
};
int PyOp({0})::py_init(PyObject *self, PyObject *args, PyObject *kwds) {{
    {2}
}

void _init_py_{0}(py::module m) {{
    using py_op = PyOp({0});
    auto& py_type = PyOpType({0});
    py_type = {{PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.{0}";
    py_type.tp_basicsize = sizeof(PyOp({0}));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "{0}";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_getset = py_op::py_getsetters;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    {3}
    PyType_Modified(&py_type);
    m.add_object("{0}", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace({0}::typeinfo(), &py_type).second);
}
)",
    op.getCppClassName(), llvm::join(getsetters, "\n    "), initBody, body);
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
    EnumContext ctx;
    using namespace std::placeholders;
    for_each_operator(os, keeper,
        std::bind(gen_op_def_pybind11_single, _1, _2, std::ref(ctx)));
    return false;
}

static bool gen_op_def_python_c_extension(raw_ostream &os, RecordKeeper &keeper) {
    EnumContext ctx;
    using namespace std::placeholders;
    for_each_operator(os, keeper,
        std::bind(gen_op_def_python_c_extension_single, _1, _2, std::ref(ctx)));
    os << "#define INIT_ALL_OP(m)";
    for_each_operator(os, keeper, [&](raw_ostream& os, MgbOp& op) {
        os << formatv(" \\\n    _init_py_{0}(m);", op.getCppClassName());
    });
    os << "\n";
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
    if (action == ActionType::CPython) {
        return TableGenMain(argv[0], &gen_op_def_python_c_extension);
    }
    return -1;
}
