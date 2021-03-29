/**
 * \file imperative/tablegen/emitter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <unordered_map>
#include <stdexcept>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tblgen {

struct Environment {
    std::unordered_map<unsigned int, std::pair<llvm::StringRef, llvm::StringRef>> enumAlias;
};

struct EmitterBase {
    EmitterBase(raw_ostream& os_): os(os_) {}
    EmitterBase(raw_ostream& os_, Environment& env): os(os_), env_p(&env) {}
protected:
    void newline() { os << "\n"; }
    Environment& env() {
        if (env_p) {
            return *env_p;
        }
        throw std::runtime_error("access global environment via non-environment emitter");
    }
    raw_ostream& os;
    Environment* env_p = nullptr;
};

} // namespace mlir::tblgen