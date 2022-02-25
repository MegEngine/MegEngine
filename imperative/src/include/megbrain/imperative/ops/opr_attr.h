#pragma once

#include "megbrain/imperative/op_def.h"
#include "megbrain/opr/param_defs.h"

namespace mgb {
namespace imperative {

struct OprAttr : public OpDefImplBase<OprAttr> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Type = std::string;
    struct Param : public std::vector<char> {
        template <typename T>
        void write_pod(const T& data) {
            static_assert(
                    !std::is_pointer<T>::value && is_location_invariant<T>::value);
            const char* ptr = static_cast<const char*>(static_cast<const void*>(&data));
            insert(end(), ptr, ptr + sizeof(T));
        }
        template <typename T, typename... Args>
        void write_pod(const T& data, const Args&... args) {
            write_pod(data);
            write_pod(args...);
        }
    };

    Type type;
    Param param;
    megdnn::param::ExecutionPolicy policy;
    cg::OperatorNodeConfig config;

    OprAttr() = default;
    OprAttr(const Type& t) : type(t) {}
    OprAttr(const Type& t, const Param& p, const cg::OperatorNodeConfig& c)
            : type(t), param(p), config(c) {}
    OprAttr(const Type& t, const Param& p, const megdnn::param::ExecutionPolicy ps,
            const cg::OperatorNodeConfig& c)
            : type(t), param(p), policy(ps), config(c) {}

    std::string repr() const;

    bool is_same_st(const Hashable& rhs) const override;
    size_t hash() const override;
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
