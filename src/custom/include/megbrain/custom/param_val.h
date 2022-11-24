#pragma once

#include <cassert>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "utils.h"

namespace custom {

/**
 * we can add a new basic data type here, basic means we can perform binary
 * op such as: +, -, *, /, ==, != between any two of them
 */
// clang-format off
#define CUSTOM_FOR_EACH_BASIC_PARAMTYPE(cb, ...)    \
    cb(Int32, int32_t, ##__VA_ARGS__)               \
    cb(Int64, int64_t, ##__VA_ARGS__)               \
    cb(Uint32, uint32_t, ##__VA_ARGS__)             \
    cb(Uint64, uint64_t, ##__VA_ARGS__)             \
    cb(Float32, float, ##__VA_ARGS__)               \
    cb(Float64, double, ##__VA_ARGS__)              \
    cb(Bool, bool, ##__VA_ARGS__)
// clang-format on

#define CUSTOM_FOR_STRING_PARAMTYPE(cb, ...) cb(String, std::string, ##__VA_ARGS__)

// clang-format off
#define CUSTOM_FOR_EACH_BASIC_LIST_PARAMTYPE(cb, ...)       \
    cb(Int32List, std::vector<int32_t>, ##__VA_ARGS__)      \
    cb(Int64List, std::vector<int64_t>, ##__VA_ARGS__)      \
    cb(Uint32List, std::vector<uint32_t>, ##__VA_ARGS__)    \
    cb(Uint64List, std::vector<uint64_t>, ##__VA_ARGS__)    \
    cb(Float32List, std::vector<float>, ##__VA_ARGS__)      \
    cb(Float64List, std::vector<double>, ##__VA_ARGS__)
// clang-format on

#define CUSTOM_FOR_BOOL_LIST_PARAMTYPE(cb, ...) \
    cb(BoolList, std::vector<bool>, ##__VA_ARGS__)

#define CUSTOM_FOR_STRING_LIST_PARAMTYPE(cb, ...) \
    cb(StringList, std::vector<std::string>, ##__VA_ARGS__)

/**
 * to avoid the recursive of MACRO
 */
// clang-format off
#define CUSTOM_FOR_EACH_BASIC_PARAMTYPE_COPY(cb, ...)   \
    cb(Int32, int32_t, ##__VA_ARGS__)                   \
    cb(Int64, int64_t, ##__VA_ARGS__)                   \
    cb(Uint32, uint32_t, ##__VA_ARGS__)                 \
    cb(Uint64, uint64_t, ##__VA_ARGS__)                 \
    cb(Float32, float, ##__VA_ARGS__)                   \
    cb(Float64, double, ##__VA_ARGS__)                  \
    cb(Bool, bool, ##__VA_ARGS__)
// clang-format on

class Device;

#define CUSTOM_FOR_EACH_VALID_PARAMTYPE(cb, ...)            \
    CUSTOM_FOR_EACH_BASIC_PARAMTYPE(cb, ##__VA_ARGS__)      \
    CUSTOM_FOR_STRING_PARAMTYPE(cb, ##__VA_ARGS__)          \
    CUSTOM_FOR_EACH_BASIC_LIST_PARAMTYPE(cb, ##__VA_ARGS__) \
    CUSTOM_FOR_BOOL_LIST_PARAMTYPE(cb, ##__VA_ARGS__)       \
    CUSTOM_FOR_STRING_LIST_PARAMTYPE(cb, ##__VA_ARGS__)     \
    cb(Device, ::custom::Device, ##__VA_ARGS__)

#define CUSTOM_FOR_EACH_LIST_PARAMTYPE(cb, ...)             \
    CUSTOM_FOR_EACH_BASIC_LIST_PARAMTYPE(cb, ##__VA_ARGS__) \
    CUSTOM_FOR_BOOL_LIST_PARAMTYPE(cb, ##__VA_ARGS__)       \
    CUSTOM_FOR_STRING_LIST_PARAMTYPE(cb, ##__VA_ARGS__)

/**
 * Macro Callback for Register
 */
#define CUSTOM_REG_DYN_PARAMTYPE(dyn_type, static_type) dyn_type,
#define CUSTOM_REG_DYN_PARAMTYPE_NAME(dyn_type, static_type) \
    {ParamDynType::dyn_type, #dyn_type},

#define CUSTOM_REG_DYN_PARAMTYPE_GETTER(dyn_type, static_type)       \
    template <>                                                      \
    struct get_dyn_type<static_type> {                               \
        static constexpr ParamDynType type = ParamDynType::dyn_type; \
    };

#define CUSTOM_REG_STATIC_PARAMTYPE_GETTER(dyn_type, static_type) \
    template <>                                                   \
    struct get_static_type<ParamDynType::dyn_type> {              \
        using type = static_type;                                 \
    };

enum class ParamDynType : uint32_t {
    CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_REG_DYN_PARAMTYPE) Invalid = 255
};

static std::unordered_map<
        ParamDynType, std::string, EnumHash<ParamDynType>, EnumCmp<ParamDynType>>
        type2name = {CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_REG_DYN_PARAMTYPE_NAME){
                ParamDynType::Invalid, "Invalid"}};

/**
 * get the dynamic data type according to the builtin static data type
 * we can use it like:
 *     ParamDynType dyn_type = get_dyn_type<int32_t>::type;
 *     assert(dyn_type == ParamDynType::Int32)
 */
template <typename T>
struct get_dyn_type {
    static constexpr ParamDynType type = ParamDynType::Invalid;
};

/**
 * get the static data type according to the dynamic data type
 * we can use it like:
 *     get_static_type<ParamDynType::Int32>::type int_32_value;
 *     assert(std::is_same<decltype(int_32_value), int>::value)
 */
template <ParamDynType>
struct get_static_type;

CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_REG_DYN_PARAMTYPE_GETTER)
CUSTOM_FOR_EACH_VALID_PARAMTYPE(CUSTOM_REG_STATIC_PARAMTYPE_GETTER)

#undef CUSTOM_REG_DYN_PARAMTYPE
#undef CUSTOM_REG_DYN_PARAMTYPE_NAME
#undef CUSTOM_REG_DYN_PARAMTYPE_GETTER
#undef CUSTOM_REG_STATIC_PARAMTYPE_GETTER

template <typename T>
struct get_vector_template_arg_type;

template <typename T>
struct get_vector_template_arg_type<std::vector<T>> {
    using type = std::decay_t<T>;
};

template <typename T>
struct is_vector {
    static constexpr bool value = false;
};

template <typename T>
struct is_vector<std::vector<T>> {
    static constexpr bool value = true;
};

template <typename T>
std::string vec2str(const std::vector<T>& vec) {
    std::stringstream ss;
    ss << "{";
    for (const auto& val : vec) {
        ss << val << ", ";
    }
    if (vec.size() != 0) {
        ss.seekp(ss.tellp() - std::streampos(2));
    }
    ss << "}";
    return ss.str();
}

/**
 * we use void* rather than template to help us realise a complete dynamic type
 * if we use template such as:
 *   template <typename T>
 *   class ParamVal {
 *       T m_data;
 *   }
 * Con1: user need to set the type explicitly when class template instantiation
 * Con2: ParamVal<int> can not be assigned to ParamVal<double>
 */
class MGE_WIN_DECLSPEC_FUC ParamVal {
    std::unique_ptr<void, void_deleter> m_ptr;
    ParamDynType m_type;

public:
    template <typename T>
    ParamVal(const T& val);
    template <typename T>
    ParamVal(const std::initializer_list<T>& val);

    ParamVal();
    ParamVal(const char* str);
    ParamVal(const std::initializer_list<const char*>& strs);
    ParamVal(const std::vector<const char*>& strs);
    ParamVal(const ParamVal& rhs);

    template <typename T>
    ParamVal& operator=(const T& rhs);
    template <typename T>
    ParamVal& operator=(const std::initializer_list<T>& val);

    ParamVal& operator=(const char* str);
    ParamVal& operator=(const std::initializer_list<const char*>& strs);
    ParamVal& operator=(const std::vector<const char*>& strs);
    ParamVal& operator=(const ParamVal& rhs);

    template <typename T>
    const T& as(void) const;
    template <typename T>
    T& as(void);

    const void* raw_ptr(void) const;
    void* raw_ptr(void);
    ParamDynType type(void) const;
    std::string str(void) const;
    size_t size(void) const;

    static std::string to_bytes(const ParamVal& value);
    static ParamVal from_bytes(const std::string& bytes, size_t& offset);

    MGE_WIN_DECLSPEC_FUC friend ParamVal operator+(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend ParamVal operator-(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend ParamVal operator*(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend ParamVal operator/(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator==(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator!=(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator>(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator<(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator>=(
            const ParamVal& lhs, const ParamVal& rhs);
    MGE_WIN_DECLSPEC_FUC friend bool operator<=(
            const ParamVal& lhs, const ParamVal& rhs);
};

ParamVal operator+(const ParamVal& lhs, const ParamVal& rhs);
ParamVal operator-(const ParamVal& lhs, const ParamVal& rhs);
ParamVal operator*(const ParamVal& lhs, const ParamVal& rhs);
ParamVal operator/(const ParamVal& lhs, const ParamVal& rhs);
bool operator==(const ParamVal& lhs, const ParamVal& rhs);
bool operator!=(const ParamVal& lhs, const ParamVal& rhs);
bool operator>(const ParamVal& lhs, const ParamVal& rhs);
bool operator<(const ParamVal& lhs, const ParamVal& rhs);
bool operator>=(const ParamVal& lhs, const ParamVal& rhs);
bool operator<=(const ParamVal& lhs, const ParamVal& rhs);

template <typename T>
ParamVal::ParamVal(const T& val) : m_ptr(nullptr, impl_deleter<std::decay_t<T>>) {
    using DecayType = std::decay_t<T>;
    m_type = get_dyn_type<DecayType>::type;
    custom_assert(
            m_type != ParamDynType::Invalid,
            "param construct error! unsupported builtin type");
    m_ptr.reset(new DecayType(val));
}

template <typename T>
ParamVal::ParamVal(const std::initializer_list<T>& val)
        : ParamVal(std::vector<std::decay_t<T>>(val)) {}

template <typename T>
ParamVal& ParamVal::operator=(const T& rhs) {
    using DecayType = std::decay_t<T>;
    ParamDynType rhs_dyn_type = get_dyn_type<DecayType>::type;
    custom_assert(rhs_dyn_type != ParamDynType::Invalid, "unsupported builtin dtype");

    if (rhs_dyn_type == m_type) {
        TypedRef(DecayType, m_ptr.get()) = rhs;
    } else {
        m_type = rhs_dyn_type;
        std::unique_ptr<void, void_deleter> new_ptr(
                new DecayType(rhs), impl_deleter<DecayType>);
        m_ptr.swap(new_ptr);
    }
    return *this;
}

template <typename T>
ParamVal& ParamVal::operator=(const std::initializer_list<T>& val) {
    return this->operator=(std::vector<std::decay_t<T>>(val));
}

template <typename T>
const T& ParamVal::as(void) const {
    return const_cast<ParamVal*>(this)->as<T>();
}

template <typename T>
T& ParamVal::as(void) {
    using DecayType = std::decay_t<T>;
    ParamDynType t_dyn_type = get_dyn_type<DecayType>::type;
    custom_assert(
            t_dyn_type == m_type, "type mismatch, type %s cannot be cast to type %s\n",
            type2name[m_type].c_str(), type2name[t_dyn_type].c_str());
    return TypedRef(T, m_ptr.get());
}

}  // namespace custom
