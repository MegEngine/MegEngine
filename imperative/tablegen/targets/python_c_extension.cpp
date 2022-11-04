#include <cctype>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../emitter.h"
#include "python_c_extension.h"

namespace mlir::tblgen {
namespace {

class TypeInfo;
std::pair<TypeInfo, int> parse_type(const std::string&, const int);
std::pair<std::vector<std::string>, int> parse_namespace(const std::string&, const int);

struct Unit {};
Unit unit;

struct ParseError {};

class TypeInfo {
public:
    TypeInfo(std::string name) : name(name) {}

    std::string to_python_type_string() {
        std::stringstream ss;
        ss << translate_type_name(name);
        if (params.size() > 0) {
            ss << "[" << params[0].to_python_type_string();
            for (auto i = 1; i < params.size(); i++) {
                ss << ", " << params[i].to_python_type_string();
            }
            ss << "]";
        }
        return ss.str();
    }

    std::string translate_type_name(const std::string& cppTypeName) {
        auto res = translation.find(cppTypeName);
        if (res != translation.end())
            return res->second;
        try {
            auto segments = parse_namespace(cppTypeName, 0).first;
            // special rules
            if (segments.size() > 3 && segments[0] == "megdnn" &&
                segments[1] == "param") {
                segments.erase(segments.begin(), segments.begin() + 3);
            } else if (
                    segments.size() == 2 && segments[0] == "megdnn" &&
                    segments[1] == "DType") {
                segments.erase(segments.begin(), segments.begin() + 1);
                segments[0] = "str";
            } else if (
                    segments.size() == 2 && segments[0] == "mgb" &&
                    segments[1] == "CompNode") {
                segments.erase(segments.begin(), segments.begin() + 1);
                segments[0] = "str";
            }
            std::stringstream joined;
            joined << segments[0];
            for (auto i = 1; i < segments.size(); i++) {
                joined << "." << segments[i];
            }
            return joined.str();
        } catch (ParseError) {
            return cppTypeName;
        }
    }

    std::string name;
    std::vector<TypeInfo> params;

private:
    static const std::unordered_map<std::string, std::string> translation;
};

const std::unordered_map<std::string, std::string> TypeInfo::translation = {
        {"bool", "bool"},       {"double", "float"},     {"float", "float"},
        {"int32_t", "int"},     {"int8_t", "int"},       {"size_t", "int"},
        {"std::string", "str"}, {"std::tuple", "tuple"}, {"std::vector", "list"},
        {"uint32_t", "int"},    {"uint64_t", "int"},
};

// a parser takes:
//   1. a string to parse
//   2. location to parse from (index of character)
// returns:
//   1. parsing result (type T)
//   2. end location of substring which is consumed by parsing
// throws exception when failed to parse
template <typename T>
using Parser = std::function<std::pair<T, int>(const std::string&, const int)>;

std::pair<Unit, int> parse_blank(const std::string& text, const int begin) {
    auto now = begin;
    while (now < text.length() && isblank(text[now]))
        now += 1;
    return {unit, now};
}

Parser<Unit> parse_non_blank_char(char ch) {
    return [=](const std::string& text, const int begin) -> std::pair<Unit, int> {
        auto blankEnd = parse_blank(text, begin).second;
        if (blankEnd >= text.length() || text[blankEnd] != ch)
            throw ParseError{};
        return {unit, blankEnd + 1};
    };
}

Parser<std::string> parse_allowed_chars(std::function<bool(char)> allow) {
    return [=](const std::string& text,
               const int begin) -> std::pair<std::string, int> {
        auto now = begin;
        while (now < text.length() && allow(text[now]))
            now += 1;
        return {text.substr(begin, now - begin), now};
    };
}

template <typename T>
Parser<std::tuple<T>> parse_seq(Parser<T> only) {
    return [=](const std::string& text,
               const int begin) -> std::pair<std::tuple<T>, int> {
        auto res = only(text, begin);
        return {{res.first}, res.second};
    };
}

template <typename Head, typename... Tail>
Parser<std::tuple<Head, Tail...>> parse_seq(Parser<Head> head, Parser<Tail>... tail) {
    return [=](const std::string& text,
               const int begin) -> std::pair<std::tuple<Head, Tail...>, int> {
        std::pair<Head, int> headRes = head(text, begin);
        std::pair<std::tuple<Tail...>, int> tailRes =
                parse_seq(tail...)(text, headRes.second);
        return {std::tuple_cat(std::tuple<Head>(headRes.first), tailRes.first),
                tailRes.second};
    };
}

template <typename T>
Parser<std::vector<T>> parse_many_at_least0(Parser<T> one) {
    return [=](const std::string& text,
               const int begin) -> std::pair<std::vector<T>, int> {
        std::vector<T> ret;
        auto now = begin;
        try {
            while (true) {
                auto oneRes = one(text, now);
                ret.emplace_back(oneRes.first);
                now = oneRes.second;
            }
        } catch (ParseError) {
        }
        return {ret, now};
    };
}

template <typename C>
Parser<std::vector<C>> parse_sep_by_at_least1(
        Parser<Unit> separator, Parser<C> component) {
    return [=](const std::string& text,
               const int begin) -> std::pair<std::vector<C>, int> {
        std::vector<C> ret;
        auto headRes = component(text, begin);
        ret.emplace_back(headRes.first);
        auto tailRes = parse_many_at_least0(parse_seq(separator, component))(
                text, headRes.second);
        for (const auto& elem : tailRes.first) {
            ret.emplace_back(std::get<1>(elem));
        }
        return {ret, tailRes.second};
    };
}

std::pair<std::string, int> parse_identifier(const std::string& text, const int begin) {
    auto blankEnd = parse_blank(text, begin).second;
    auto indentRes = parse_allowed_chars(
            [](char ch) { return std::isalnum(ch) || ch == '_'; })(text, blankEnd);
    if (indentRes.first.empty())
        throw ParseError{};
    return indentRes;
};

std::pair<std::string, int> parse_qualified(const std::string& text, const int begin) {
    auto blankEnd = parse_blank(text, begin).second;
    auto indentRes = parse_allowed_chars([](char ch) {
        return std::isalnum(ch) || ch == '_' || ch == ':';
    })(text, blankEnd);
    if (indentRes.first.empty())
        throw ParseError{};
    return indentRes;
};

std::pair<std::vector<std::string>, int> parse_namespace(
        const std::string& text, const int begin) {
    auto res = parse_many_at_least0(parse_seq(
            parse_non_blank_char(':'), parse_non_blank_char(':'),
            Parser<std::string>(parse_identifier)))(text, begin);
    std::vector<std::string> ret;
    for (const auto& elem : res.first) {
        ret.emplace_back(std::get<2>(elem));
    }
    return {ret, res.second};
}

std::pair<TypeInfo, int> parse_leaf_type(const std::string& text, const int begin) {
    auto ret = parse_qualified(text, begin);
    return {TypeInfo(ret.first), ret.second};
};

std::pair<TypeInfo, int> parse_node_type(const std::string& text, const int begin) {
    auto nameRes = parse_qualified(text, begin);
    auto ret = TypeInfo(nameRes.first);
    auto now = parse_non_blank_char('<')(text, nameRes.second).second;
    auto argsRes = parse_sep_by_at_least1(
            parse_non_blank_char(','), Parser<TypeInfo>(parse_type))(text, now);
    ret.params = argsRes.first;
    now = parse_non_blank_char('>')(text, argsRes.second).second;
    return {ret, now};
};

std::pair<TypeInfo, int> parse_type(const std::string& text, const int begin) {
    try {
        return parse_node_type(text, begin);
    } catch (ParseError) {
    }
    return parse_leaf_type(text, begin);
};

std::string cpp_type_to_python_type(const std::string& input) {
    auto res = parse_type(input, 0);
    return res.first.to_python_type_string();
}

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
    void emit_py_init_proxy();
    void emit_py_init_methoddef(
            const std::unordered_map<std::string, std::vector<std::string>>&
                    enum_attr_members);
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
    std::unordered_map<std::string, std::vector<std::string>> enum_attr_members;

    for (auto&& i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            subclasses.push_back(
                    EnumAttrEmitter(op.getCppClassName(), attr, os, env()).emit());

            auto retType = cpp_type_to_python_type(std::string(attr->getReturnType()));
            enum_attr_members[retType] = std::vector<std::string>();
            for (const auto& member : attr->getEnumMembers()) {
                enum_attr_members[retType].emplace_back(member);
            }
        }
    }

    emit_class();
    emit_py_init();
    emit_py_getsetters();
    emit_py_methods();
    emit_py_init_proxy();
    emit_py_init_methoddef(enum_attr_members);
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
    static PyObject* py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds);
    static PyMethodDef py_init_methoddef;
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

void OpDefEmitter::emit_py_init_proxy() {
    os << tgfmt(
            R"(
PyObject *PyOp($_self)::py_init_proxy(PyObject *self, PyObject *args, PyObject *kwds) {
    if (PyOp($_self)::py_init(self, args, kwds) < 0) {
        return NULL;
    }
    Py_RETURN_NONE;
}
)",
            &ctx);
}

void OpDefEmitter::emit_py_init_methoddef(
        const std::unordered_map<std::string, std::vector<std::string>>&
                enum_attr_members) {
    std::string docstring = "__init__(self";
    for (const auto& attr : op.getMgbAttributes()) {
        if (attr.name == "workspace_limit")
            continue;
        auto pyType = cpp_type_to_python_type(std::string(attr.attr.getReturnType()));
        auto findRes = enum_attr_members.find(pyType);
        if (findRes != enum_attr_members.end()) {
            pyType = formatv("Union[str, {0}]", pyType);
            // TODO stubgen cannot handle Literal strings for now
            // auto members = findRes->second;
            // std::string enumTypeString = "Literal[";
            // enumTypeString += formatv("'{0}'", lowercase(members[0]));
            // for (auto i = 1; i < members.size(); i++) {
            //     enumTypeString += formatv(", '{0}'", lowercase(members[i]));
            // }
            // enumTypeString += "]";
            // pyType = enumTypeString;
        }
        docstring += formatv(", {0}: {1} = ...", attr.name, pyType);
    }
    docstring += ") -> None\\n";
    os << tgfmt(
            R"(
PyMethodDef PyOp($_self)::py_init_methoddef = {
    "__init__",
    (PyCFunction)PyOp($_self)::py_init_proxy,
    METH_VARARGS | METH_KEYWORDS,
    "$0"
};
)",
            &ctx, docstring);
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

    py_type.tp_dict = PyDict_New();
    PyObject* descr = PyDescr_NewMethod(&PyOpType($_self), &PyOp($_self)::py_init_methoddef);
    PyDict_SetItemString(py_type.tp_dict, "__init__", descr);
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
