/**
 * \file imperative/tablegen/targets/pybind11.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./pybind11.h"
#include "../emitter.h"

namespace mlir::tblgen {
namespace {
class OpDefEmitter final: public EmitterBase {
public:
    OpDefEmitter(MgbOp& op_, raw_ostream& os_, Environment& env_):
        EmitterBase(os_, env_), op(op_) {}

    void emit();
private:
    MgbOp& op;
};

void OpDefEmitter::emit() {
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
            auto&& enumAlias = env().enumAlias;
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
                if (attr->getEnumCombinedFlag()) {
                    //! define operator |
                    os << formatv(
                            "\n    .def(\"__or__\", []({0}::{1} s0, {0}::{1} s1) {{ "
                            "\n         return static_cast<{0}::{1}>(uint32_t(s0) | uint32_t(s1));"
                            "\n      })",
                            className, attr->getEnumName());
                    //! define operator &
                    os << formatv(
                            "\n    .def(\"__and__\", []({0}::{1} s0, {0}::{1} s1) {{"
                            "\n         return static_cast<{0}::{1}>(uint32_t(s0) & uint32_t(s1));"
                            "\n    })",
                            className, attr->getEnumName());
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
        os << ", std::string>()";
        for (auto &&i : op.getMgbAttributes()) {
            os << formatv(", py::arg(\"{0}\")", i.name);
            auto defaultValue = i.attr.getDefaultValue();
            if (!defaultValue.empty()) {
                os << formatv(" = {0}", defaultValue);
            } else {
                hasDefaultCtor = true;
            }
        }
        os << ", py::arg(\"scope\") = {})";
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
} // namespace

bool gen_op_def_pybind11(raw_ostream &os, llvm::RecordKeeper &keeper) {
    Environment env;
    using namespace std::placeholders;
    foreach_operator(keeper, [&](MgbOp& op) {
        OpDefEmitter(op, os, env).emit();
    });
    return false;
}
} // namespace mlir::tblgen
