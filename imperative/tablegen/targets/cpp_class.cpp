/**
 * \file imperative/tablegen/targets/cpp_class.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cpp_class.h"
#include "../emitter.h"

namespace mlir::tblgen {
namespace {
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

class OpDefEmitter final: public EmitterBase {
public:
    OpDefEmitter(MgbOp& op_, raw_ostream& os_):
        EmitterBase(os_), op(op_) {}
    void emit_header();
    void emit_tpl_spl();
    void emit_body();
private:
    MgbOp& op;
};

void OpDefEmitter::emit_header() {
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
        paramList.push_back("std::string scope_ = {}");
        gen_ctor(llvm::join(paramList, ", "),
                 ": " + llvm::join(initList, ", "),
                 " { set_scope(scope_); }");
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

void OpDefEmitter::emit_tpl_spl() {
    for (auto &&i : op.getMgbAttributes()) {
        if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
            if (attr->supportToString()) {
                std::vector<std::string> case_body;
                std::string ename = formatv("{0}::{1}",
                    op.getCppClassName(), attr->getEnumName());
                llvm::for_each(attr->getEnumMembers(), [&](auto&& v){
                    case_body.push_back(formatv(
                        "case {0}::{1}: return \"{1}\";", ename, v));
                });
                os << formatv(R"(
template <>
struct ToStringTrait<{0}> {
    std::string operator()({0} e) const {
        switch (e) {
            {1}
            default:
                return "{0}::Unknown";
        }
    }
};
)", ename, llvm::join(case_body, "\n"));
            }
        }
    }
}

void OpDefEmitter::emit_body() {
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

        // generate props()
        os << formatv(
            "std::vector<std::pair<const char*, std::string>> {0}(const OpDef& def_) {{\n",
            formatMethImpl("props")
        );
        os << formatv(
            "    auto&& op_ = def_.cast_final_safe<{0}>();\n"
            "    static_cast<void>(op_);\n",
            className
        );
        ctx.withSelf("op_");
        os << mlir::tblgen::tgfmt(hashable->getPropsFunctionTemplate(), &ctx);
        os << "}\n";

        // generate make_name()
        os << formatv(
            "std::string {0}(const OpDef& def_) {{\n", formatMethImpl("make_name")
        );
        os << formatv(
            "    auto&& op_ = def_.cast_final_safe<{0}>();\n"
            "    static_cast<void>(op_);\n",
            className
        );
        ctx.withSelf("op_");
        os << mlir::tblgen::tgfmt(op.getNameFunctionTemplate(), &ctx);
        os << "}\n";

        os << "} // anonymous namespace\n";

        methods.push_back("hash");
        methods.push_back("is_same_st");
        methods.push_back("props");
        methods.push_back("make_name");
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
} // namespace

bool gen_op_def_c_header(raw_ostream &os, llvm::RecordKeeper &keeper) {
    foreach_operator(keeper, [&](MgbOp& op) {
        OpDefEmitter emitter(op, os);
        emitter.emit_header();
        emitter.emit_tpl_spl();
    });
    return false;
}

bool gen_op_def_c_body(raw_ostream &os, llvm::RecordKeeper &keeper) {
    foreach_operator(keeper, [&](MgbOp& op) {
        OpDefEmitter emitter(op, os);
        emitter.emit_body();
    });
    return false;
}
} // namespace mlir::tblgen
