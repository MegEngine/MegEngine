#pragma once

#include <stdexcept>
#include <unordered_map>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tblgen {
using llvm::raw_ostream;

struct Environment {
    std::unordered_map<unsigned int, std::pair<llvm::StringRef, llvm::StringRef>>
            enumAlias;
};

struct EmitterBase {
    EmitterBase(raw_ostream& os_) : os(os_) {}
    EmitterBase(raw_ostream& os_, Environment& env) : os(os_), env_p(&env) {}

protected:
    void newline() { os << "\n"; }
    Environment& env() {
        if (env_p) {
            return *env_p;
        }
        throw std::runtime_error(
                "access global environment via non-environment emitter");
    }
    raw_ostream& os;
    Environment* env_p = nullptr;
};

}  // namespace mlir::tblgen
