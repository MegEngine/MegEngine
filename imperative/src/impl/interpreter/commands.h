#pragma once

#include <string>
#include <unordered_set>
#include <variant>

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/utils/to_string.h"
#include "megbrain/tensor.h"

#include "./stack_manager.h"
#include "./tensor_info.h"

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
        // functor("value", value);
    }

    const char* get_name() const { return "Put"; }
};

struct ApplyOp {
    uint64_t id;  // used by profiler to identify unique apply
    std::shared_ptr<OpDef> op;
    SmallVector<TensorInfo*> inputs;
    SmallVector<TensorInfo*> outputs;
    bool validated = false;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("op", op);
        functor("inputs", inputs);
        functor("outputs", outputs);
    }

    const char* get_name() const { return "ApplyOp"; }
};

struct Del {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const { return "Del"; }
};

struct GetValue {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const { return "GetValue"; }
};

struct Drop {
    TensorInfo* dest;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("dest", dest);
    }

    const char* get_name() const { return "Drop"; }
};

struct SetOption {
    std::string key;
    size_t value;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("key", key);
        functor("value", value);
    }

    const char* get_name() const { return "SetOption"; }
};

struct StartProfile {
    std::unordered_set<TensorInfo*> capture_tensors;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {}

    const char* get_name() const { return "StartProfile"; }
};

struct StopProfile {
    std::unordered_set<TensorInfo*> escape_tensors;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {}

    const char* get_name() const { return "StopProfile"; }
};

struct PushScope {
    std::string scope_name;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("scope_name", scope_name);
    }

    const char* get_name() const { return "PushScope"; }
};

struct PopScope {
    std::string scope_name;

    template <typename TFunctor>
    void get_props(TFunctor&& functor) const {
        functor("scope_name", scope_name);
    }

    const char* get_name() const { return "PopScope"; }
};

using CommandData = std::variant<
        Put, ApplyOp, Del, GetValue, Drop, SetOption, StartProfile, StopProfile,
        PushScope, PopScope>;

struct Command {
    uint64_t id;
    CommandData data;
    StackManager::Trace trace;
};
// using IdentifiedCommand = std::pair<uint64_t, Command>;

}  // namespace interpreter::intl

template <>
struct ToStringTrait<interpreter::intl::Command> {
    std::string operator()(const interpreter::intl::Command& cmd) const {
        std::string content = std::visit(
                [](const auto& cmd) {
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
                },
                cmd.data);
        return content;
    }
};

}  // namespace mgb::imperative
