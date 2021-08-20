/**
 * \file imperative/tablegen/targets/macros.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./macros.h"
#include "./cpp_class.h"
#include "../emitter.h"

namespace mlir::tblgen {
bool gen_enum_param_list_macro(raw_ostream &os, llvm::RecordKeeper &keeper) {
    std::vector<std::pair<std::string, std::string>> enums;
    std::vector<std::pair<std::string, std::string>> bit_enums;
    Environment env;
    foreach_operator(keeper, [&](MgbOp& op) {
        for (auto&& i : op.getAttributes()) {
            if (auto attr = llvm::dyn_cast<MgbEnumAttr>(&i.attr)) {
                auto insert = [&](const MgbEnumAttr& attr) {
                    auto&& item = std::make_pair(
                            attr.getParentNamespace(), attr.getEnumName());
                    if (env.enumAlias.emplace(
                            attr.getBaseRecord()->getID(), std::move(item)).second) {
                        if (attr.getEnumCombinedFlag()) {
                            bit_enums.emplace_back(item);
                        } else {
                            enums.emplace_back(item);
                        }
                    }
                };
                if (auto alias = llvm::dyn_cast<MgbAliasAttr>(attr)) {
                    auto&& aliasBase = alias->getAliasBase();
                    insert(llvm::cast<MgbEnumAttr>(aliasBase));
                } else {
                    insert(*attr);
                }
            }
        }
    });
    os << "#define FOR_EACH_ENUM_PARAM(cb)";
    for (auto && i : enums) {
        os << formatv(" \\\n    cb({0}::{1});", i.first, i.second);
    }
    os << "\n";
    os << "#define FOR_EACH_BIT_COMBINED_ENUM_PARAM(cb)";
    for (auto && i : bit_enums) {
        os << formatv(" \\\n    cb({0}::{1});", i.first, i.second);
    }
    os << "\n";
    return false;
}
} // namespace mlir::tblgen
