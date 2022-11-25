#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include "./basic_operators.h"
#include "./operator.h"
#include "./value.h"
#include "megbrain/common.h"

namespace mgb::imperative {

struct BackTraceInfo;
using BackTraceInfoPtr = std::shared_ptr<BackTraceInfo>;

struct PyFrameInfo {
    virtual std::string traceback() = 0;
    virtual ~PyFrameInfo() {}
};
using PyFrameInfoPtr = std::shared_ptr<PyFrameInfo>;

using OpAttrInfo = std::variant<std::monostate, std::string, GetAttr::Attr>;

struct TransformationCallInfo {
    size_t depth;
    std::string op;
    std::string transform;
    OpAttrInfo attrs;
    std::vector<std::string> inp_types;
    std::string to_string() {
        static const char tabs[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
        const char* prefix = tabs + (sizeof(tabs) / sizeof(char)) - depth - 1;
        std::string inps = "";
        for (auto i : inp_types) {
            inps += i + ", ";
        }
        std::string opinfo = op;
        std::visit(
                [&opinfo](auto&& i) {
                    using T = std::decay_t<decltype(i)>;
                    if constexpr (std::is_same_v<T, std::string>) {
                        opinfo += "(" + i + ")";
                    } else if constexpr (std::is_same_v<T, GetAttr::Attr>) {
                        switch (i) {
                            case GetAttr::Attr::Data:
                                opinfo += "(data)";
                                break;
                            case GetAttr::Attr::Shape:
                                opinfo += "(shape)";
                                break;
                            case GetAttr::Attr::DType:
                                opinfo += "(dtype)";
                                break;
                            case GetAttr::Attr::Device:
                                opinfo += "(device)";
                                break;
                            case GetAttr::Attr::Value:
                                opinfo += "(value)";
                                break;
                            case GetAttr::Attr::None:
                                opinfo += "(none)";
                                break;
                            default:
                                break;
                        }
                    }
                },
                attrs);
        return ssprintf(
                "%s %s: Apply (%s, %s)", prefix, transform.c_str(), opinfo.c_str(),
                inps.c_str());
    }

    static OpAttrInfo get_op_attr(const Operator& op) {
        if (op.is<GetAttr>()) {
            return op.as<GetAttr>()->attr();
        } else if (op.is<ApplyOp>()) {
            auto& opdef = op.as<ApplyOp>()->op();
            return opdef.name();
        } else {
            return {};
        }
    }
};

struct TransformationReturnInfo {
    size_t depth;
    std::vector<std::string> return_types;

    std::string to_string() {
        static const char tabs[] = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";
        const char* prefix = tabs + (sizeof(tabs) / sizeof(char)) - depth - 1;
        std::string returns = "";
        for (auto i : return_types) {
            returns += i + ", ";
        }
        return ssprintf("%s return: %s", prefix, returns.c_str());
    }
};

struct BackTraceInfo {
    std::vector<std::variant<TransformationCallInfo, TransformationReturnInfo>>
            trans_stack_info;
    PyFrameInfoPtr py_stack_info;

    BackTraceInfo(PyFrameInfoPtr info) : py_stack_info{std::move(info)} {}

    std::string py_traceback() {
        return "Python Backtrace: " + py_stack_info->traceback();
    }

    std::string transformation_traceback() {
        std::string trace_info = "Dispatch Transformation Backtrace: ";
        for (auto&& i : trans_stack_info) {
            std::visit(
                    [&trace_info](auto& i) { trace_info += "\n" + i.to_string(); }, i);
        }
        return trace_info;
    }
};

}  // namespace mgb::imperative