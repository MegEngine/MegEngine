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
    Initproc(std::string&& s) : func(std::move(s)) {}
    std::string operator()(std::string argument) {
        return formatv("{0}({1})", func, argument);
    }
};

class OpDefEmitter : public EmitterBase {
public:
    OpDefEmitter(MgbOp& op_, raw_ostream& os_, Environment& env_)
            : EmitterBase(os_, env_), op(op_) {
        ctx.withSelf(op.getCppClassName());
    }

    Initproc emit();

private:
    void emit_class();
    void emit_py_init();
    void emit_py_getsetters();
    void emit_py_methods();
    Initproc emit_initproc();

    MgbOp& op;
    std::vector<Initproc> subclasses;
    mlir::tblgen::FmtContext ctx;
};

class EnumAttrEmitter : public EmitterBase {
public:
    EnumAttrEmitter(
            llvm::StringRef parent, MgbEnumAttr* attr_, raw_ostream& os_,
            Environment& env_)
            : EmitterBase(os_, env_), attr(attr_) {
        unsigned int enumID;
        if (auto alias = llvm::dyn_cast<MgbAliasAttr>(attr)) {
            auto&& aliasBase = alias->getAliasBase();
            enumID = llvm::cast<MgbEnumAttr>(aliasBase).getBaseRecord()->getID();
        } else {
            enumID = attr->getBaseRecord()->getID();
        }
        ctx.addSubst(
                "enumTpl",
                attr->getEnumCombinedFlag() ? "BitCombinedEnumWrapper" : "EnumWrapper");
        ctx.addSubst("opClass", parent);
        ctx.addSubst("enumClass", attr->getEnumName());
        firstOccur =
                env().enumAlias
                        .emplace(enumID, std::make_pair(parent, attr->getEnumName()))
                        .second;
    }

    Initproc emit();

protected:
    void emit_trait();
    void emit_tpl_spl();
    Initproc emit_initproc();

    MgbEnumAttr* attr;
    bool firstOccur;
    mlir::tblgen::FmtContext ctx;
};

Initproc EnumAttrEmitter::emit() {
    emit_trait();
    emit_tpl_spl();
    return emit_initproc();
}

void EnumAttrEmitter::emit_trait() {
    if (!firstOccur)
        return;

    auto enumMax = [&] {
        if (attr->getEnumCombinedFlag()) {
            return formatv("(1llu << {0}) - 1", attr->getEnumMembers().size());
        } else {
            return formatv("{0} - 1", attr->getEnumMembers().size());
        }
    };
    os << tgfmt(
            R"(
template<> struct EnumTrait<$opClass::$enumClass> {
    static constexpr const char *name = "$opClass.$enumClass";
    static constexpr std::underlying_type_t<$opClass::$enumClass> max = $0;
};
)",
            &ctx, enumMax());
}

void EnumAttrEmitter::emit_tpl_spl() {
    if (!firstOccur)
        return;

    os << tgfmt(
            "template<> PyTypeObject* $enumTpl<$opClass::$enumClass>::type = "
            "nullptr;\n",
            &ctx);

    auto quote = [&](auto&& i) -> std::string {
        size_t d1 = i.find(' ');
        size_t d2 = i.find('=');
        size_t d = d1 <= d2 ? d1 : d2;
        return formatv("\"{0}\"", i.substr(0, d));
    };
    os << tgfmt(
            R"(
template<> const char*
$enumTpl<$opClass::$enumClass>::members[] = {$0};
)",
            &ctx, llvm::join(llvm::map_range(attr->getEnumMembers(), quote), ", "));

    auto mem2value = [&](auto&& i) -> std::string {
        size_t d1 = i.find(' ');
        size_t d2 = i.find('=');
        size_t d = d1 <= d2 ? d1 : d2;
        return tgfmt(
                "{normalize_enum(\"$0\"), $opClass::$enumClass::$0}", &ctx,
                i.substr(0, d));
    };
    os << tgfmt(
            R"(
template<> std::unordered_map<std::string, $opClass::$enumClass>
$enumTpl<$opClass::$enumClass>::mem2value = {$0};
)",
            &ctx, llvm::join(llvm::map_range(attr->getEnumMembers(), mem2value), ", "));

    os << tgfmt(
            "template<> PyObject* "
            "$enumTpl<$opClass::$enumClass>::pyobj_insts[$0] = {nullptr};\n",
            &ctx, attr->getEnumMembers().size());
}

Initproc EnumAttrEmitter::emit_initproc() {
    std::string initproc =
            formatv("_init_py_{0}_{1}", ctx.getSubstFor("opClass"),
                    ctx.getSubstFor("enumClass"));

    os << tgfmt(
            R"(
void $0(PyTypeObject& py_type) {
    auto& e_type = $enumTpl<$opClass::$enumClass>::type;
)",
            &ctx, initproc);

    if (firstOccur) {
        os << tgfmt(
                R"(
    static PyMethodDef tp_methods[] = {
        {const_cast<char*>("dump"), (PyCFunction)$enumTpl<$opClass::$enumClass>::py_dump, METH_NOARGS, NULL},
        {NULL}  /* Sentinel */
        };
    )",
                &ctx);
        os << tgfmt(
                R"(
    static PyType_Slot slots[] = {
        {Py_tp_repr, (void*)$enumTpl<$opClass::$enumClass>::py_repr},
        {Py_tp_richcompare, (void*)$enumTpl<$opClass::$enumClass>::tp_richcompare},
        {Py_tp_methods, tp_methods},
)",
                &ctx);
        if (attr->getEnumCombinedFlag()) {
            // only bit combined enum could new instance because bitwise operation,
            // others should always use singleton
            os << tgfmt(
                    R"(
        {Py_tp_new, (void*)$enumTpl<$opClass::$enumClass>::py_new_combined_enum},
        {Py_nb_or, (void*)$enumTpl<$opClass::$enumClass>::py_or},
        {Py_nb_and, (void*)$enumTpl<$opClass::$enumClass>::py_and},
)",
                    &ctx);
        }
        os << R"(
        {0, NULL}
    };)";

        os << tgfmt(
                R"(
    static PyType_Spec spec = {
        // name
        "megengine.core._imperative_rt.ops.$opClass.$enumClass",
        // basicsize
        sizeof($enumTpl<$opClass::$enumClass>),
        // itemsize
        0,
        // flags
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
        // slots
        slots
    };)",
                &ctx);

        os << tgfmt(
                R"(
    e_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));
)",
                &ctx);

        for (auto&& i :
             {std::pair<std::string, std::string>{
                      "__name__", tgfmt("$enumClass", &ctx)},
              {"__module__", "megengine.core._imperative_rt.ops"},
              {"__qualname__", tgfmt("$opClass.$enumClass", &ctx)}}) {
            os << formatv(
                    R"(
    mgb_assert(
            e_type->tp_setattro(
                    reinterpret_cast<PyObject*>(e_type),
                    py::cast("{0}").release().ptr(),
                    py::cast("{1}").release().ptr()) >= 0);
)",
                    i.first, i.second);
        }

        auto&& members = attr->getEnumMembers();
        for (size_t idx = 0; idx < members.size(); ++idx) {
            size_t d1 = members[idx].find(' ');
            size_t d2 = members[idx].find('=');
            size_t d = d1 <= d2 ? d1 : d2;
            os << tgfmt(
                    R"({
    PyObject* inst = e_type->tp_alloc(e_type, 0);
    reinterpret_cast<$enumTpl<$opClass::$enumClass>*>(inst)->value = $opClass::$enumClass::$0;
    mgb_assert(PyDict_SetItemString(e_type->tp_dict, "$0", inst) >= 0);
    $enumTpl<$opClass::$enumClass>::pyobj_insts[$1] = inst;
})",
                    &ctx, members[idx].substr(0, d), idx);
        }
    }

    os << tgfmt(
            R"(
    Py_INCREF(e_type);
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "$enumClass", reinterpret_cast<PyObject*>(e_type)) >= 0);
)",
            &ctx);
    os << "}\n";
    return initproc;
}

Initproc OpDefEmitter::emit() {
    for (auto&& i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            subclasses.push_back(
                    EnumAttrEmitter(op.getCppClassName(), attr, os, env()).emit());
        }
    }

    emit_class();
    emit_py_init();
    emit_py_getsetters();
    emit_py_methods();
    return emit_initproc();
}

void OpDefEmitter::emit_class() {
    auto&& className = op.getCppClassName();
    std::string method_defs;
    std::vector<std::string> body;

    llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
        body.push_back(
                formatv(R"(
            {{"{0}", serialization<decltype(opdef.{0})>::dump(opdef.{0})})",
                        attr.name));
    });
    method_defs +=
            formatv(R"(
    static PyObject* getstate(PyObject* self, PyObject*) {{
        auto& opdef = reinterpret_cast<PyOp({0})*>(self)->inst();
        static_cast<void>(opdef);
        std::unordered_map<std::string, py::object> state {{
            {1}
        };
        return py::cast(state).release().ptr();
    })",
                    className, llvm::join(body, ","));

    body.clear();
    llvm::for_each(op.getMgbAttributes(), [&](auto&& attr) {
        body.push_back(
                formatv(R"(
        {{
        auto&& iter = state.find("{0}");
        if (iter != state.end()) {
            opdef.{0} = serialization<decltype(opdef.{0})>::load(iter->second);
        }
        })",
                        attr.name));
    });

    method_defs +=
            formatv(R"(
    static PyObject* setstate(PyObject* self, PyObject* args) {{
        PyObject* dict = PyTuple_GetItem(args, 0);
        if (!dict) return NULL;
        auto state = py::cast<std::unordered_map<std::string, py::object>>(dict);
        auto& opdef = reinterpret_cast<PyOp({0})*>(self)->inst();
        static_cast<void>(opdef);
        {1}
        Py_RETURN_NONE;
    })",
                    className, llvm::join(body, "\n"));

    os << tgfmt(
            R"(
PyOpDefBegin($_self) // {
    static PyGetSetDef py_getsetters[];
    static PyMethodDef tp_methods[];
    $0
    static int py_init(PyObject *self, PyObject *args, PyObject *kwds);
// };
PyOpDefEnd($_self)
)",
            &ctx, method_defs);
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
        initBody +=
                llvm::join(llvm::map_range(attr_name_list, initializer), ", ") + ";\n";
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
            initBody +=
                    tgfmt(R"(
    if ($0) {
        try {
            // TODO: remove this guard which is used for pybind11 implicit conversion
            py::detail::loader_life_support guard{};
            reinterpret_cast<PyOp($_self)*>(self)->inst().$0 =
                    py::cast<decltype($_self::$0)>(py::handle($0));
        } CATCH_ALL(-1)
    }
)",
                          &ctx, attr.name);
        });

        initBody +=
                tgfmt(R"(
    if (scope) {
        try {
            reinterpret_cast<PyOp(OpDef)*>(self)->op
                ->set_scope(py::cast<std::string>(py::handle(scope)));
        } CATCH_ALL(-1)
    }
)",
                      &ctx);
    }
    initBody += "\n    return 0;";

    os << tgfmt(
            R"(
int PyOp($_self)::py_init(PyObject *self, PyObject *args, PyObject *kwds) {
    $0
}
)",
            &ctx, initBody);
}

void OpDefEmitter::emit_py_getsetters() {
    auto f = [&](auto&& attr) -> std::string {
        return tgfmt(
                "{const_cast<char*>(\"$0\"), py_get_generic($_self, $0), "
                "py_set_generic($_self, $0), const_cast<char*>(\"$0\"), NULL},",
                &ctx, attr.name);
    };
    os << tgfmt(
            R"(
PyGetSetDef PyOp($_self)::py_getsetters[] = {
    $0
    {NULL}  /* Sentinel */
};
)",
            &ctx, llvm::join(llvm::map_range(op.getMgbAttributes(), f), "\n    "));
}

void OpDefEmitter::emit_py_methods() {
    // generate methods
    std::string method_defs;
    std::vector<std::string> method_items;
    {
        auto&& className = op.getCppClassName();
        // generate getstate
        method_items.push_back(
                formatv("{{const_cast<char*>(\"__getstate__\"), PyOp({0})::getstate, "
                        "METH_NOARGS, \"{0} getstate\"},",
                        className));

        // generate setstate
        method_items.push_back(
                formatv("{{const_cast<char*>(\"__setstate__\"), PyOp({0})::setstate, "
                        "METH_VARARGS, \"{0} setstate\"},",
                        className));
    }

    os << tgfmt(
            R"(
    PyMethodDef PyOp($_self)::tp_methods[] = {
        $0
        {NULL}  /* Sentinel */
    };
    )",
            &ctx, llvm::join(method_items, "\n    "));
}

Initproc OpDefEmitter::emit_initproc() {
    std::string initproc = formatv("_init_py_{0}", op.getCppClassName());
    std::string subclass_init_call;
    for (auto&& i : subclasses) {
        subclass_init_call += formatv("    {0};\n", i("py_type"));
    }
    os << tgfmt(
            R"(
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
    py_type.tp_methods = py_op::tp_methods;
    py_type.tp_getset = py_op::py_getsetters;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    $1
    PyType_Modified(&py_type);
    m.add_object("$_self", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace($_self::typeinfo(), &py_type).second);
}
)",
            &ctx, initproc, subclass_init_call);
    return initproc;
}
}  // namespace

bool gen_op_def_python_c_extension(raw_ostream& os, llvm::RecordKeeper& keeper) {
    Environment env;
    using namespace std::placeholders;
    std::vector<Initproc> initprocs;
    foreach_operator(keeper, [&](MgbOp& op) {
        initprocs.emplace_back(OpDefEmitter(op, os, env).emit());
    });
    os << "#define INIT_ALL_OP(m)";
    for (auto&& init : initprocs) {
        os << formatv(" \\\n    {0};", init("m"));
    }
    os << "\n";
    return false;
}
}  // namespace mlir::tblgen