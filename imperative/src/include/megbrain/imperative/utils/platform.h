#pragma once

#include <string>
#include <typeinfo>
namespace mgb::imperative {

std::string demangle(std::string mangled);

template <typename T>
const char* demangled_typename() {
    static auto name = demangle(typeid(T).name());
    return name.c_str();
}

}  // namespace mgb::imperative
