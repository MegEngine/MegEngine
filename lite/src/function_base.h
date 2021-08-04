/**
 * \file src/function_base.h
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
#include "misc.h"
#include "type_info.h"
// template <typename tensor_type, typename ...Arg>
namespace lite {
class TensorImplDft;
class NetworkImplDft;
namespace {

template <typename class_type>
struct class_type_name {
    std::string operator()() { return ""; }
};
#define ADD_STATEMENT(class_name, backend_name)            \
    template <>                                            \
    struct class_type_name<class_name> {                   \
        std::string operator()() { return #backend_name; } \
    }
ADD_STATEMENT(TensorImplDft, Dft);
ADD_STATEMENT(NetworkImplDft, Dft);
#undef ADD_STATEMENT
}  // namespace

// if it can't find the function, ignore
template <typename tensor_type, typename ret_type, typename... Args>
ret_type try_call_func(std::string func_name, Args... args) {
    mark_used_variable(func_name);
    mark_used_variable(args...);
    return nullptr;
}

// if it can't find the function, throw error
template <typename tensor_type, typename ret_type, typename... Args>
ret_type call_func(std::string func_name, Args... args) {
    mark_used_variable(args...);
    auto backend_name = class_type_name<tensor_type>()();
    auto msg_info =
            func_name + "  is not aviliable in " + backend_name + " backend.";
    LITE_THROW(msg_info.c_str());
}
}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
