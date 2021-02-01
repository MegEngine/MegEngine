/**
 * \file imperative/src/impl/interpreter/commands.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <string>
#include <variant>

#include "megbrain/tensor.h"
#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/utils/to_string.h"

namespace mgb::imperative {

namespace interpreter::intl {

struct TensorInfo;
class InterpreterProfiler;

struct Put {
    TensorInfo* dest;
    HostTensorND value;
    bool no_cache = false;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
        functor("no_cache", no_cache);
        //functor("value", value);
    }

    const char* get_name() const {
        return "Put";
    }
};

struct ApplyOp {
    std::shared_ptr<OpDef> op;
    SmallVector<TensorInfo*> inputs;
    SmallVector<TensorInfo*> outputs;
    SmallVector<TensorInfo*> dels;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("op", op);
        functor("inputs", inputs);
        functor("outputs", outputs);
        functor("dels", dels);
    }

    const char* get_name() const {
        return "ApplyOp";
    }
};

struct Del {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const {
        return "Del";
    }
};

struct GetValue {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const {
        return "GetValue";
    }
};

struct SwapIn {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const {
        return "SwapIn";
    }
};

struct SwapOut {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const {
        return "SwapOut";
    }
};

struct Drop {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const {
        return "Drop";
    }
};

struct SetOption {
    std::string key;
    int value;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("key", key);
        functor("value", value);
    }

    const char* get_name() const {
        return "SetOption";
    }
};

struct StartProfile {
    InterpreterProfiler* profiler;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {}

    const char* get_name() const {
        return "StartProfile";
    }
};

struct StopProfile {
    std::string basename;
    std::string format;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("basename", basename);
        functor("format", format);
    }

    const char* get_name() const {
        return "StopProfile";
    }
};

struct PushScope {
    std::string scope_name;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("scope_name", scope_name);
    }

    const char* get_name() const {
        return "PushScope";
    }
};

struct PopScope {
    std::string scope_name;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("scope_name", scope_name);
    }

    const char* get_name() const {
        return "PopScope";
    }
};

using Command = std::variant<Put,
                             ApplyOp,
                             Del,
                             GetValue,
                             SwapIn,
                             SwapOut,
                             Drop,
                             SetOption,
                             StartProfile,
                             StopProfile,
                             PushScope,
                             PopScope>;

using IdentifiedCommand = std::pair<uint64_t, Command>;

}

template <>
struct ToStringTrait<interpreter::intl::Command>{
    std::string operator()(const interpreter::intl::Command& cmd) const {
        return std::visit([](const auto& cmd){
            std::string result = cmd.get_name();
            result += "{";
            cmd.get_props([&](const char* key, auto&& value) {
                result += key;
                result += ": ";
                result += to_string(value);
                result += ",";
            });
            result += "}";
            return result;
        }, cmd);
    }
};

}
