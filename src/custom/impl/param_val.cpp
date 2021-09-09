/**
 * \file src/custom/impl/param_val.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/custom/param_val.h"
#include "megbrain/common.h"

#pragma GCC diagnostic ignored "-Wsign-compare"

using namespace mgb;

namespace custom {

/**
 * Macro Callback for Case
 */

#define CUSTOM_CASE_TO_ALLOC_ACCORD_TO_RHS(dyn_type, static_type)                   \
    case (ParamDynType::dyn_type): {                                                \
        std::unique_ptr<void, void_deleter> new_ptr(                                \
            new static_type(TypedRef(static_type, rhs.m_ptr.get())),                \
            impl_deleter<static_type>                                               \
        );                                                                          \
        m_ptr.swap(new_ptr);                                                        \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_ASSIGN_ACCORD_TO_RHS(dyn_type, static_type)                  \
    case (ParamDynType::dyn_type): {                                                \
        TypedRef(static_type, m_ptr.get()) = TypedRef(static_type, rhs.m_ptr.get());\
        break;                                                                      \
    }

#define CUSTOM_ASSERT_OPERAND_VALID(operand, opr)                                   \
    mgb_assert(                                                                     \
        operand.m_ptr != nullptr && operand.m_type != ParamDynType::Invalid,        \
        "invalid %s of operator %s of ParamVal", #operand, #opr                     \
    )

#define CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op)                                      \
    mgb_assert(                                                                     \
        lhs.m_type == rhs.m_type, "`%s` %s `%s` is not allowed",                    \
        type2name[lhs.m_type].c_str(), #op,                                         \
        type2name[rhs.m_type].c_str()                                               \
    )

#define CUSTOM_CASE_TO_GET_BINARY_OP_RHS_AND_CAL(dyn_type, static_type, op)         \
    case (ParamDynType::dyn_type): {                                                \
        const auto &rval = TypedRef(static_type, rhs.m_ptr.get());                  \
        return lval op rval;                                                        \
    }

#define CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_BASIC(dyn_type, static_type, op)           \
    case (ParamDynType::dyn_type): {                                                \
        const auto &lval = TypedRef(static_type, lhs.m_ptr.get());                  \
        switch (rhs.m_type) {                                                       \
            CUSTOM_FOR_EACH_BASIC_PARAMTYPE_COPY(                                   \
                CUSTOM_CASE_TO_GET_BINARY_OP_RHS_AND_CAL, op)                       \
            default:                                                                \
                CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op);                             \
        }                                                                           \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_NONBASIC(dyn_type, static_type, op)        \
    case (ParamDynType::dyn_type): {                                                \
        CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op);                                     \
        const auto &lval = TypedRef(static_type, lhs.m_ptr.get());                  \
        const auto &rval = TypedRef(static_type, rhs.m_ptr.get());                  \
        return lval op rval;                                                        \
    }

#define CUSTOM_DEFINE_BINARY_OP_FOR_BASIC(op, ret_type)                             \
    ret_type operator op(const ParamVal &lhs, const ParamVal &rhs) {                \
        CUSTOM_ASSERT_OPERAND_VALID(lhs, op);                                       \
        CUSTOM_ASSERT_OPERAND_VALID(rhs, op);                                       \
                                                                                    \
        switch (lhs.m_type) {                                                       \
            CUSTOM_FOR_EACH_BASIC_PARAMTYPE(                                        \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_BASIC, op)                         \
            default:                                                                \
                CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op);                             \
        }                                                                           \
        return {};                                                                  \
    }

#define CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING(op, ret_type)                  \
    ret_type operator op(const ParamVal &lhs, const ParamVal &rhs) {                \
        CUSTOM_ASSERT_OPERAND_VALID(lhs, op);                                       \
        CUSTOM_ASSERT_OPERAND_VALID(rhs, op);                                       \
                                                                                    \
        switch (lhs.m_type) {                                                       \
            CUSTOM_FOR_EACH_BASIC_PARAMTYPE(                                        \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_BASIC, op)                         \
            CUSTOM_FOR_STRING_PARAMTYPE(                                            \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_NONBASIC, op)                      \
            default:                                                                \
                CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op);                             \
        }                                                                           \
        return {};                                                                  \
    }

#define CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(op, ret_type)         \
    ret_type operator op(const ParamVal &lhs, const ParamVal &rhs) {                \
        CUSTOM_ASSERT_OPERAND_VALID(lhs, op);                                       \
        CUSTOM_ASSERT_OPERAND_VALID(rhs, op);                                       \
                                                                                    \
        switch (lhs.m_type) {                                                       \
            CUSTOM_FOR_EACH_BASIC_PARAMTYPE(                                        \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_BASIC, op)                         \
            CUSTOM_FOR_STRING_PARAMTYPE(                                            \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_NONBASIC, op)                      \
            CUSTOM_FOR_EACH_LIST_PARAMTYPE(                                         \
                CUSTOM_CASE_TO_CAL_BINARY_OP_FOR_NONBASIC, op)                      \
            default:                                                                \
                CUSTOM_INVALID_EXPR_EXCP(lhs, rhs, op);                             \
        }                                                                           \
        return {};                                                                  \
    }

#define CUSTOM_CASE_TO_PRINT_NONLIST(dyn_type, static_type)                         \
    case (ParamDynType::dyn_type): {                                                \
        auto rval = TypedRef(static_type, m_ptr.get());                             \
        ss << rval;                                                                 \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_PRINT_LIST(dyn_type, static_type)                            \
    case (ParamDynType::dyn_type): {                                                \
        auto rval = TypedRef(static_type, m_ptr.get());                             \
        ss << vec2str(rval);                                                        \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_RET_SIZE(dyn_type, static_type)                              \
    case (ParamDynType::dyn_type): {                                                \
        return TypedRef(static_type, m_ptr.get()).size();                           \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_DUMP_BASIC(dyn_type, static_type)                            \
    case (ParamDynType::dyn_type): {                                                \
        res.resize(sizeof(ParamDynType) + sizeof(static_type));                     \
        memcpy(&res[0], &(value.m_type), sizeof(ParamDynType));                     \
        memcpy(&res[sizeof(ParamDynType)], value.m_ptr.get(), sizeof(static_type)); \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_DUMP_LIST(dyn_type, static_type)                             \
    case (ParamDynType::dyn_type): {                                                \
        auto &ref = TypedRef(static_type, value.m_ptr.get());                       \
        size_t len = ref.size();                                                    \
        size_t elem_size = len != 0 ? sizeof(ref[0]) : 0;                           \
        res.resize(sizeof(ParamDynType) + sizeof(len) + len*elem_size);             \
        memcpy(&res[0], &(value.m_type), sizeof(ParamDynType));                     \
        memcpy(&res[sizeof(ParamDynType)], &len, sizeof(len));                      \
        memcpy(&res[sizeof(ParamDynType)+sizeof(len)], ref.data(), len*elem_size);  \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_LOAD_BASIC(dyn_type, static_type)                            \
    case (ParamDynType::dyn_type): {                                                \
        static_type val;                                                            \
        memcpy(&val, &bytes[offset], sizeof(val));                                  \
        offset += sizeof(val);                                                      \
        return val;                                                                 \
        break;                                                                      \
    }

#define CUSTOM_CASE_TO_LOAD_LIST(dyn_type, static_type)                             \
    case (ParamDynType::dyn_type): {                                                \
        size_t len = 0;                                                             \
        memcpy(&len, &bytes[offset], sizeof(len));                                  \
        offset += sizeof(len);                                                      \
        static_type vals;                                                           \
        vals.resize(len);                                                           \
        size_t elem_size = len != 0 ? sizeof(vals[0]) : 0;                          \
        memcpy(&vals[0], &bytes[offset], len*elem_size);                            \
        offset += len*elem_size;                                                    \
        return vals;                                                                \
        break;                                                                      \
    }

ParamVal::ParamVal(): m_ptr(nullptr, [](void*) -> void {}) {
    m_type = ParamDynType::Invalid;
}

ParamVal::ParamVal(const char *str): ParamVal(std::string(str)) {

}

ParamVal::ParamVal(const std::initializer_list<const char*> &strs): ParamVal(std::vector<const char*>(strs)) {
}

ParamVal::ParamVal(const std::vector<const char*> &strs)
        : m_ptr(new std::vector<std::string>(), impl_deleter<std::vector<std::string>>) {
    m_type = ParamDynType::StringList;
    for (const auto &str: strs) {
        TypedRef(std::vector<std::string>, m_ptr.get()).emplace_back(str);
    }
}

ParamVal::ParamVal(const ParamVal &rhs): m_ptr(nullptr, [](void*) -> void {}) {
    mgb_assert(
        rhs.m_type != ParamDynType::Invalid && rhs.m_ptr != nullptr,
        "invalid rhs of copy constructor of ParamVal"
    );
    m_type = rhs.m_type;
    switch(m_type) {
        CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_CASE_TO_ALLOC_ACCORD_TO_RHS)
        default: {
            mgb_assert(false, "invalid rhs of copy constructor of ParamVal");
        }
    }
}

ParamVal &ParamVal::operator=(const char *str) {
    this->operator=(std::string(str));
    return *this;
}

ParamVal &ParamVal::operator=(const std::initializer_list<const char*> &strs) {
    this->operator=(std::vector<const char*>(strs));
    return *this;
}

ParamVal &ParamVal::operator=(const std::vector<const char*> &strs) {
    std::vector<std::string> tmp_strs;
    for (const auto &str: strs) {
        tmp_strs.emplace_back(str);
    }
    this->operator=(tmp_strs);
    return *this;
}

ParamVal &ParamVal::operator=(const ParamVal &rhs) {
    if (&rhs == this)
        return *this;
    mgb_assert(
        rhs.m_type != ParamDynType::Invalid && rhs.m_ptr != nullptr,
        "invalid rhs of assignment operator of ParamVal"
    );
    
    if (rhs.m_type == m_type) {
        switch(m_type) {
            CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_CASE_TO_ASSIGN_ACCORD_TO_RHS);
            default:
                mgb_assert(false, "invalid rhs of assignment operator of ParamVal");
        }
    }
    else {
        m_type = rhs.m_type;
        switch(m_type) {
            CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_CASE_TO_ALLOC_ACCORD_TO_RHS);
            default:
                mgb_assert(false, "invalid rhs of assignment operator of ParamVal");
        }
    }
    return *this;
}

const void *ParamVal::raw_ptr(void) const {
    return m_ptr.get();
}

void *ParamVal::raw_ptr(void) {
    return m_ptr.get();
}

ParamDynType ParamVal::type(void) const {
    return m_type;
}

std::string ParamVal::str() const {
    std::stringstream ss;
    ss << "type: " << type2name[m_type] << "\n" << "value: ";
    switch (m_type) {
        CUSTOM_FOR_EACH_BASIC_PARAMTYPE(CUSTOM_CASE_TO_PRINT_NONLIST)
        CUSTOM_FOR_STRING_PARAMTYPE(CUSTOM_CASE_TO_PRINT_NONLIST)
        CUSTOM_FOR_EACH_LIST_PARAMTYPE(CUSTOM_CASE_TO_PRINT_LIST)
        default:
            mgb_assert(false, "invalid data of assignment operator of ParamVal");
    }
    return ss.str();
}

size_t ParamVal::size(void) const {
    switch (m_type) {
        CUSTOM_FOR_STRING_PARAMTYPE(CUSTOM_CASE_TO_RET_SIZE)
        CUSTOM_FOR_EACH_LIST_PARAMTYPE(CUSTOM_CASE_TO_RET_SIZE)
        default:
            mgb_assert(false, "there is no size() for basic data types");
    }
}

std::string ParamVal::to_bytes(const ParamVal &value) {
    std::string res;
    // because the specialization of std::vector<bool>
    if (value.type() == ParamDynType::BoolList) {
        std::vector<bool> &ref = TypedRef(std::vector<bool>, value.m_ptr.get());
        size_t len = ref.size();
        size_t elem_size = sizeof(bool);
        res.resize(sizeof(ParamDynType) + sizeof(len) + len*elem_size);
        memcpy(&res[0], &(value.m_type), sizeof(ParamDynType));
        memcpy(&res[sizeof(ParamDynType)], &len, sizeof(len));
        size_t startpos = sizeof(ParamDynType)+sizeof(len);
        for (size_t idx=0; idx<len; idx++) {
            bool b = ref[idx];
            memcpy(&res[startpos+idx*sizeof(b)], &b, sizeof(b));
        }
        return res;
    }
    else if (value.type() == ParamDynType::StringList) {
        std::vector<std::string> &ref = TypedRef(std::vector<std::string>, value.m_ptr.get());
        size_t len = ref.size();
        res.resize(sizeof(ParamDynType) + sizeof(len));
        memcpy(&res[0], &(value.m_type), sizeof(ParamDynType));
        memcpy(&res[sizeof(ParamDynType)], &len, sizeof(len));
        for (size_t idx=0; idx<ref.size(); ++idx) {
            size_t str_len = ref[idx].size();
            std::string bytes(sizeof(str_len) + str_len, ' ');
            memcpy(&bytes[0], &str_len, sizeof(str_len));
            memcpy(&bytes[sizeof(str_len)], ref[idx].data(), str_len);
            res += bytes;
        }
        return res;
    }
    switch(value.type()) {
        CUSTOM_FOR_EACH_BASIC_PARAMTYPE(CUSTOM_CASE_TO_DUMP_BASIC)
        CUSTOM_FOR_STRING_PARAMTYPE(CUSTOM_CASE_TO_DUMP_LIST)
        CUSTOM_FOR_EACH_BASIC_LIST_PARAMTYPE(CUSTOM_CASE_TO_DUMP_LIST)
        default:
            mgb_assert(false, "invalid param type");
    }
    return res;
}

ParamVal ParamVal::from_bytes(const std::string &bytes, size_t &offset) {
    ParamDynType data_type = ParamDynType::Invalid;
    memcpy(&data_type, &bytes[offset], sizeof(ParamDynType));
    offset += sizeof(ParamDynType);
    if (data_type == ParamDynType::BoolList) {
        std::vector<bool> ret;
        size_t len = 0;
        memcpy(&len, &bytes[offset], sizeof(len));
        offset += sizeof(len);
        for (size_t idx =0; idx<len; ++idx) {
            bool b = true;
            memcpy(&b, &bytes[offset], sizeof(bool));
            offset += sizeof(bool);
            ret.push_back(b);
        }
        return ret;
    }
    else if (data_type == ParamDynType::StringList) {
        std::vector<std::string> ret;
        size_t len = 0;
        memcpy(&len, &bytes[offset], sizeof(len));
        offset += sizeof(len);
        for (size_t idx =0; idx<len; ++idx) {
            size_t str_len = 0;
            memcpy(&str_len, &bytes[offset], sizeof(str_len));
            offset += sizeof(str_len);
            std::string str(str_len, ' ');
            memcpy(&str[0], &bytes[offset], str_len);
            offset += str_len;
            ret.push_back(str);
        }
        return ret;
    }

    switch (data_type) {
        CUSTOM_FOR_EACH_BASIC_PARAMTYPE(CUSTOM_CASE_TO_LOAD_BASIC)
        CUSTOM_FOR_STRING_PARAMTYPE(CUSTOM_CASE_TO_LOAD_LIST)
        CUSTOM_FOR_EACH_BASIC_LIST_PARAMTYPE(CUSTOM_CASE_TO_LOAD_LIST);
        default:
            mgb_assert(false, "invalid param type");
    }
    return {};
}

CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING(+, ParamVal)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC(-, ParamVal)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC(*, ParamVal)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC(/, ParamVal)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(==, bool)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(!=, bool)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(>=, bool)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(<=, bool)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(>, bool)
CUSTOM_DEFINE_BINARY_OP_FOR_BASIC_AND_STRING_AND_LIST(<, bool)

}
