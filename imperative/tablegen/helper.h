/**
 * \file imperative/tablegen/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"

using llvm::formatv;
using llvm::StringRef;
using llvm::Record;

#define ASSERT(stmt, msg) \
    if (!(stmt)) { \
        std::cerr << "\033[1;31m" \
            << "tablegen autogen abort due to: " << msg \
            << "\033[0m" << std::endl; \
        exit(1); \
    }

namespace mlir {
namespace tblgen {
template<typename ConcreteType>
struct MgbInterface : public ConcreteType {
    MgbInterface() = delete;
    MgbInterface(const MgbInterface&) = delete;
    MgbInterface(MgbInterface&&) = delete;
    ~MgbInterface() = delete;
};

struct MgbAttrWrapperBase : public MgbInterface<Attribute> {
private:
    struct RecordVisitor : public MgbInterface<Constraint> {
    public:
        static bool classof(const Constraint*) {
            return true;
        }

        const llvm::Record* getDef() const {
            return def;
        }
    };
public:
    static bool classof(const Attribute* attr) {
        return attr->isSubClassOf("MgbAttrWrapperBase");
    }

    const llvm::Record* getBaseRecord() const {
        auto baseAttr = getBaseAttr();
        return llvm::cast<RecordVisitor>(baseAttr).getDef();
    }
    llvm::StringRef getUnderlyingType() const {
        return def->getValueAsString("underlyingType");
    }
};

struct MgbEnumAttrMixin : public MgbAttrWrapperBase {
    static bool classof(const Attribute* attr) {
        return attr->getBaseAttr().isSubClassOf("MgbEnumAttrMixin");
    }

    llvm::StringRef getParentNamespace() const {
        return getBaseRecord()->getValueAsString("parentNamespace");
    }
    llvm::StringRef getEnumName() const {
        return getBaseRecord()->getValueAsString("enumName");
    }
    std::vector<StringRef> getEnumMembers() const {
        return getBaseRecord()->getValueAsListOfStrings("enumMembers");
    }
    bool supportToString() const {
        return getBaseRecord()->getValueAsBit("supportToString");
    }
    bool getEnumCombinedFlag() const {
        return getBaseRecord()->getValueAsBit("enumCombined");
    }
};

struct MgbHashableAttrMixin : public MgbAttrWrapperBase {
    static bool classof(const Attribute* attr) {
        return attr->getBaseAttr().isSubClassOf("MgbHashableAttrMixin");
    }

    llvm::StringRef getHashFunctionTemplate() const {
        return getBaseRecord()->getValueAsString("hashFunction");
    }
    llvm::StringRef getCmpFunctionTemplate() const {
        return getBaseRecord()->getValueAsString("cmpFunction");
    }
    llvm::StringRef getReprFunctionTemplate() const {
        return getBaseRecord()->getValueAsString("reprFunction");
    }
};

struct MgbAliasAttrMixin : public MgbAttrWrapperBase {
    static bool classof(const Attribute* attr) {
        return attr->getBaseAttr().isSubClassOf("MgbAliasAttrMixin");
    }

    Attribute getAliasBase() const {
        return Attribute(getBaseRecord()->getValueAsDef("aliasBase"));
    }
};

class MgbPackedParam {
public:
    MgbPackedParam(Record* def_): def(def_) {
        auto&& dag = def->getValueAsDag("fields");
        for (size_t i = 0; i < dag->getNumArgs(); ++ i) {
            fields.push_back({
                dag->getArgNameStr(i),
                Attribute(llvm::cast<llvm::DefInit>(dag->getArg(i)))
            });
        }
    }

    llvm::StringRef getFullName() const {
        return def->getValueAsString("fullName");
    }
    std::vector<NamedAttribute> getFields() const {
        return fields;
    }
    llvm::StringRef getAccessor() const {
        return def->getValueAsString("paramAccessor");
    }
private:
    std::vector<NamedAttribute> fields;
    Record* def;
};

struct MgbOpBase : public MgbInterface<Operator> {
    static bool isPackedParam(Record* def) {
        return def->isSubClassOf("MgbPackedParamBase");
    }

public:
    static bool classof(const Operator* op) {
        return op->getDef().isSubClassOf("MgbOp");
    }

    std::vector<NamedAttribute> getMgbAttributes() const {
        std::vector<NamedAttribute> ret;
        for (auto&& i: getAttributes()) {
            if (isa<MgbAttrWrapperBase>(i.attr)) {
                ret.push_back(i);
            }
        }
        return ret;
    }
    std::vector<NamedAttribute> getExtraArguments() const {
        std::vector<NamedAttribute> ret;
        auto&& dag = getDef().getValueAsDag("extraArguments");
        for (size_t i = 0; i < dag->getNumArgs(); ++ i) {
            ret.push_back({
                dag->getArgNameStr(i),
                Attribute(llvm::cast<llvm::DefInit>(dag->getArg(i)))
            });
        }
        return ret;
    }
    llvm::Optional<StringRef> getExtraOpdefDecl() const {
        return getDef().getValueAsOptionalString("extraOpdefDecl");
    }
    std::vector<MgbPackedParam> getPackedParams() const {
        std::vector<MgbPackedParam> ret;
        for (auto&& i : getDef().getValueAsListOfDefs("dnnParams")) {
            if (isPackedParam(i)) {
                ret.emplace_back(i);
            }
        }
        return ret;
    }
    std::string getNameFunctionTemplate() const {
        if (auto f = getDef().getValueAsOptionalString("nameFunction")) {
            return f.getValue().str();
        }
        return formatv("    return \"{0}\";\n", getCppClassName());
    }
};

struct MgbHashableOpMixin : public MgbOpBase {
private:
    std::string getDefaultHashFunction() const {
        std::string body = "    size_t val = mgb::hash($_self.dyn_typeinfo());\n";
        if (!getMgbAttributes().empty()) {
            auto getHashFunc = [&](auto&& iter) {
                auto&& attr = llvm::cast<MgbHashableAttrMixin>(iter.attr);
                return attr.getHashFunctionTemplate();
            };
            mlir::tblgen::FmtContext ctx;
            for (auto&& it: getMgbAttributes()) {
                body += formatv(
                    "    val = mgb::hash_pair_combine(val, {0});\n",
                    mlir::tblgen::tgfmt(getHashFunc(it), &ctx, "$_self." + it.name)
                );
            }
        }
        body += "    return val;\n";
        return body;
    }
    std::string getDefaultCmpFunction() const {
        std::string body;
        if (!getMgbAttributes().empty()) {
            mlir::tblgen::FmtContext ctx;
            for (auto&& it : getMgbAttributes()) {
                auto&& attr = llvm::cast<MgbHashableAttrMixin>(it.attr);
                body += formatv(
                    "    if ({0}) return false;\n",
                    mlir::tblgen::tgfmt(attr.getCmpFunctionTemplate(),
                        &ctx, "$0." + it.name, "$1." + it.name)
                );
            }
        }
        body += "    return true;\n";
        return body;
    }
    std::string getDefaultPropsFunction() const {
        std::string body = "    std::vector<std::pair<const char*, std::string>> props_;\n";
        if (!getMgbAttributes().empty()) {
            mlir::tblgen::FmtContext ctx;
            for (auto&& it : getMgbAttributes()) {
                if (auto* enumAttr = llvm::dyn_cast<MgbEnumAttrMixin>(&it.attr)) {
                    body += formatv("    switch ({0}){{\n", "$_self." + it.name);
                    for (auto&& enumMember: enumAttr->getEnumMembers()) {
                        body += formatv(
                            "    case {0}::{1}::{2}:\n",
                            getCppClassName(), enumAttr->getEnumName(), enumMember
                        );
                        body += formatv(
                            "        props_.emplace_back(\"{0}\", \"{1}\");\n",
                            it.name, enumMember
                        );
                        body += "        break;\n";
                    }
                    body += "    default: break;\n";
                    body += "    }\n";
                } else {
                    auto&& attr = llvm::cast<MgbHashableAttrMixin>(it.attr);
                    body += formatv(
                        "    props_.emplace_back(\"{0}\", {1});\n", it.name,
                        mlir::tblgen::tgfmt(attr.getReprFunctionTemplate(),
                            &ctx, "$_self." + it.name)
                    );
                }
            }
        }
        body += "    return props_;\n";
        return body;
    }
public:
    static bool classof(const Operator* op) {
        return op->getDef().isSubClassOf("MgbHashableOpMixin");
    }

    std::string getHashFunctionTemplate() const {
        if (auto f = getDef().getValueAsOptionalString("hashFunction")) {
            return f.getValue().str();
        }
        return getDefaultHashFunction();
    }
    std::string getCmpFunctionTemplate() const {
        if (auto f = getDef().getValueAsOptionalString("cmpFunction")) {
            return f.getValue().str();
        }
        return getDefaultCmpFunction();
    }
    std::string getPropsFunctionTemplate() const {
        if (auto f = getDef().getValueAsOptionalString("propsFunction")) {
            return f.getValue().str();
        }
        return getDefaultPropsFunction();
    }
};

using MgbAttrWrapper = mlir::tblgen::MgbAttrWrapperBase;
using MgbEnumAttr = mlir::tblgen::MgbEnumAttrMixin;
using MgbHashableAttr = mlir::tblgen::MgbHashableAttrMixin;
using MgbAliasAttr = mlir::tblgen::MgbAliasAttrMixin;
using MgbOp = mlir::tblgen::MgbOpBase;
using MgbHashableOp = mlir::tblgen::MgbHashableOpMixin;

static inline void foreach_operator(llvm::RecordKeeper &keeper,
        std::function<void(MgbOp&)> callback) {
    auto op_base_class = keeper.getClass("Op");
    ASSERT(op_base_class, "could not find base class Op");
    for (auto&& i: keeper.getDefs()) {
        auto&& r = i.second;
        if (r->isSubClassOf(op_base_class)) {
            auto op = mlir::tblgen::Operator(r.get());
            if (op.getDialectName().str() == "mgb") {
                std::cerr << "\033[34;15m" << "Generating " << r->getName().str() << "\033[0m" << std::endl;
                callback(llvm::cast<MgbOp>(op));
            }
        }
    }
}

} // namespace tblgen
} // namespace mlir
