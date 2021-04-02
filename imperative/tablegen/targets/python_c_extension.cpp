/**
 * \file imperative/tablegen/targets/python_c_extension.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "python_c_extension.h"
#include "../emitter.h"

namespace mlir::tblgen {
namespace {
struct Initproc {
    std::string func;
    Initproc(std::string&& s): func(std::move(s)) {}
    std::string operator()(std::string argument) {
        return formatv("{0}({1})", func, argument);
    }
};

class OpDefEmitter: public EmitterBase {
public:
    OpDefEmitter(MgbOp& op_, raw_ostream& os_, Environment& env_):
        EmitterBase(os_, env_), op(op_) {
        ctx.withSelf(op.getCppClassName());
    }

    Initproc emit();
private:
    void emit_class();
    void emit_py_init();
    void emit_py_getsetters();
    Initproc emit_initproc();

    MgbOp& op;
    std::vector<Initproc> subclasses;
    mlir::tblgen::FmtContext ctx;
};

class EnumAttrEmitter: public EmitterBase {
public:
    EnumAttrEmitter(llvm::StringRef parent, MgbEnumAttr* attr_, raw_ostream& os_, Environment& env_):
        EmitterBase(os_, env_), attr(attr_) {
        unsigned int enumID;
        if (auto alias = llvm::dyn_cast<MgbAliasAttr>(attr)) {
            auto&& aliasBase = alias->getAliasBase();
            enumID = llvm::cast<MgbEnumAttr>(aliasBase).getBaseRecord()->getID();
        } else {
            enumID = attr->getBaseRecord()->getID();
        }
        ctx.addSubst("enumTpl", attr->getEnumCombinedFlag() ? "BitCombinedEnumWrapper" : "EnumWrapper");
        ctx.addSubst("opClass", parent);
        ctx.addSubst("enumClass", attr->getEnumName());
        firstOccur = env().enumAlias.emplace(enumID, std::make_pair(parent, attr->getEnumName())).second;
    }

    Initproc emit();
protected:
    void emit_tpl_spl();
    Initproc emit_initproc();

    MgbEnumAttr* attr;
    bool firstOccur;
    mlir::tblgen::FmtContext ctx;
};

Initproc EnumAttrEmitter::emit() {
    emit_tpl_spl();
    return emit_initproc();
}

void EnumAttrEmitter::emit_tpl_spl() {
    if (!firstOccur) return;

    os << tgfmt(
            "template<> PyTypeObject $enumTpl<$opClass::$enumClass>::type={};\n",
            &ctx);

    os << tgfmt(
            "template<> const char* $enumTpl<$opClass::$enumClass>::name = "
            "\"$opClass.$enumClass\";\n", 
            &ctx);

    if (attr->getEnumCombinedFlag()) {
        os << tgfmt(
                "template<> PyNumberMethods "
                "$enumTpl<$opClass::$enumClass>::number_methods={};\n",
                &ctx);
        os << tgfmt(R"(
template<> struct EnumTrait<$opClass::$enumClass> {
    static constexpr bool is_bit_combined = true;
    static constexpr std::underlying_type_t<$opClass::$enumClass> max = (1llu << $0) - 1;
};
)", &ctx, attr->getEnumMembers().size());
    }

    auto str2type = [&](auto&& i) -> std::string {
        return tgfmt("{normalize_enum(\"$0\"), $opClass::$enumClass::$0}", &ctx, i);
    };
    os << tgfmt(R"(
template<> std::unordered_map<std::string, $opClass::$enumClass>
$enumTpl<$opClass::$enumClass>::str2type = {$0};
)", &ctx, llvm::join(llvm::map_range(attr->getEnumMembers(), str2type), ", "));

    auto type2str = [&](auto&& i) -> std::string {
        return tgfmt("{$opClass::$enumClass::$0, normalize_enum(\"$0\")}", &ctx, i);
    };
    os << tgfmt(R"(
template<> std::unordered_map<$opClass::$enumClass, std::string>
$enumTpl<$opClass::$enumClass>::type2str = {$0};
)", &ctx, llvm::join(llvm::map_range(attr->getEnumMembers(), type2str), ", "));
}

Initproc EnumAttrEmitter::emit_initproc() {
    std::string initproc = formatv("_init_py_{0}_{1}",
        ctx.getSubstFor("opClass"), ctx.getSubstFor("enumClass"));

    os << tgfmt(R"(
void $0(PyTypeObject& py_type) {
    auto& e_type = $enumTpl<$opClass::$enumClass>::type;
)", &ctx, initproc);

    if (firstOccur) {
        os << tgfmt(R"(
    e_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    e_type.tp_name = "megengine.core._imperative_rt.ops.$opClass.$enumClass";
    e_type.tp_basicsize = sizeof($enumTpl<$opClass::$enumClass>);
    e_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    e_type.tp_doc = "$opClass.$enumClass";
    e_type.tp_base = &PyBaseObject_Type;
    e_type.tp_repr = $enumTpl<$opClass::$enumClass>::py_repr;
    e_type.tp_richcompare = $enumTpl<$opClass::$enumClass>::tp_richcompare;
)", &ctx);
        if (attr->getEnumCombinedFlag()) {
            // only bit combined enum could new instance because bitwise operation,
            // others should always use singleton
            os << tgfmt(R"(
    e_type.tp_new = $enumTpl<$opClass::$enumClass>::py_new_combined_enum;
    auto& number_method = $enumTpl<$opClass::$enumClass>::number_methods;
    number_method.nb_or = $enumTpl<$opClass::$enumClass>::py_or;
    number_method.nb_and = $enumTpl<$opClass::$enumClass>::py_and;
    e_type.tp_as_number = &number_method;
)", &ctx);
        }

        os << "    mgb_assert(PyType_Ready(&e_type) >= 0);\n";


        for (auto&& i : attr->getEnumMembers()) {
            os << tgfmt(R"({
    PyObject* inst = e_type.tp_alloc(&e_type, 0);
    reinterpret_cast<$enumTpl<$opClass::$enumClass>*>(inst)->value = $opClass::$enumClass::$0;
    mgb_assert(PyDict_SetItemString(e_type.tp_dict, "$0", inst) >= 0);
    PyType_Modified(&e_type);
})", &ctx, i);
        }
    }

    os << tgfmt(R"(
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "$enumClass", reinterpret_cast<PyObject*>(&e_type)) >= 0);
)", &ctx);
    os << "}\n";
    return initproc;
}   

Initproc OpDefEmitter::emit() {
    for (auto&& i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            subclasses.push_back(EnumAttrEmitter(op.getCppClassName(), attr, os, env()).emit());
        }
    }

    emit_class();
    emit_py_init();
    emit_py_getsetters();
    return emit_initproc();
}

void OpDefEmitter::emit_class() {
    os << tgfmt(R"(
PyOpDefBegin($_self) // {
    static PyGetSetDef py_getsetters[];
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
// };
PyOpDefEnd($_self)
)", &ctx);
}

void OpDefEmitter::emit_py_init() {
    std::string initBody;
    if (!op.getMgbAttributes().empty()) {
        initBody += "static const char* kwlist[] = {";

        std::vector<llvm::StringRef> attr_name_list;
        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            attr_name_list.push_back(attr.name);
        });
        attr_name_list.push_back("scope");

        llvm::for_each(attr_name_list, [&](auto&& attr) {
            initBody += formatv("\"{0}\", ", attr);
        });
        initBody += "NULL};\n";
        initBody += "    PyObject ";
        auto initializer = [&](auto&& attr) -> std::string {
            return formatv("*{0} = NULL", attr);
        };
        initBody += llvm::join(llvm::map_range(attr_name_list, initializer), ", ") + ";\n";
        initBody += "    if (!PyArg_ParseTupleAndKeywords(args, kwds, \"|";
        // an extra slot created for name
        initBody += std::string(attr_name_list.size(), 'O');
        initBody += "\", const_cast<char**>(kwlist)";
        llvm::for_each(attr_name_list, [&](auto&& attr) {
            initBody += formatv(", &{0}", attr);
        });
        initBody += "))\n";
        initBody += "    return -1;\n";

        llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
            initBody += tgfmt(R"(
    if ($0) {
        try {
            reinterpret_cast<PyOp($_self)*>(self)->inst().$0 =
                pyobj_convert_generic<decltype($_self::$0)>::from($0);
        } CATCH_ALL(-1)
    }
)", &ctx, attr.name);
        });

        initBody += tgfmt(R"(
    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(pyobj_convert_generic<std::string>::from(scope));
        } CATCH_ALL(-1)
    }
)", &ctx);

    }
    initBody += "\n    return 0;";


    os << tgfmt(R"(
int PyOp($_self)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    $0
}
)", &ctx, initBody);
}

void OpDefEmitter::emit_py_getsetters() {
    auto f = [&](auto&& attr) -> std::string {
        return tgfmt(
            "{const_cast<char*>(\"$0\"), py_get_generic($_self, $0), py_set_generic($_self, $0), const_cast<char*>(\"$0\"), NULL},",
            &ctx, attr.name);
    };
    os << tgfmt(R"(
PyGetSetDef PyOp($_self)::py_getsetters[] = {
    $0
    {NULL}  /* Sentinel */
};
)", &ctx, llvm::join(llvm::map_range(op.getMgbAttributes(), f), "\n    "));
}

Initproc OpDefEmitter::emit_initproc() {
    std::string initproc = formatv("_init_py_{0}", op.getCppClassName());
    std::string subclass_init_call;
    for (auto&& i : subclasses) {
        subclass_init_call += formatv("    {0};\n", i("py_type"));
    }
    os << tgfmt(R"(
void $0(py::module m) {
    using py_op = PyOp($_self);
    auto& py_type = PyOpType($_self);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.$_self";
    py_type.tp_basicsize = sizeof(PyOp($_self));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "$_self";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    py_type.tp_init = py_op::py_init;
    py_type.tp_getset = py_op::py_getsetters;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    $1
    PyType_Modified(&py_type);
    m.add_object("$_self", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace($_self::typeinfo(), &py_type).second);
}
)", &ctx, initproc, subclass_init_call);
    return initproc;
}
} // namespace

bool gen_op_def_python_c_extension(raw_ostream &os, llvm::RecordKeeper &keeper) {
    Environment env;
    using namespace std::placeholders;
    std::vector<Initproc> initprocs;
    foreach_operator(keeper, [&](MgbOp& op) {
        initprocs.emplace_back(OpDefEmitter(op, os, env).emit());
    });
    os << "#define INIT_ALL_OP(m)";
    for(auto&& init : initprocs) {
        os << formatv(" \\\n    {0};", init("m"));
    }
    os << "\n";
    return false;
}
} // namespace mlir::tblgen