#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/hash.h"
#include "megbrain/utils/metahelper.h"

#if MGB_ENABLE_JSON

#include <memory>
#include <unordered_map>
#include <vector>

namespace mgb {
namespace json {

class Value : public std::enable_shared_from_this<Value>, public DynTypeObj {
public:
    virtual void writeto(std::string& fout, int indent = 0) const = 0;

    MGE_WIN_DECLSPEC_FUC void writeto_fpath(
            const std::string& fout_path, int indent = 0) const {
        writeto_fpath(fout_path.c_str(), indent);
    }

    MGE_WIN_DECLSPEC_FUC void writeto_fpath(
            const char* fout_path, int indent = 0) const;

    MGE_WIN_DECLSPEC_FUC virtual std::string to_string(int indent = 0) const final;

    virtual ~Value() = default;
};

class Number final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    double m_val;

public:
    Number(double v) : m_val(v) {}

    static std::shared_ptr<Number> make(double v) {
        return std::make_shared<Number>(v);
    }

    void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class NumberInt final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    int64_t m_val;

public:
    NumberInt(int64_t v) : m_val(v) {}

    static std::shared_ptr<NumberInt> make(int64_t v) {
        return std::make_shared<NumberInt>(v);
    }

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class Bool final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    bool m_val;

public:
    Bool(bool v) : m_val(v) {}

    static std::shared_ptr<Bool> make(bool v);

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class String final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    std::string m_val;

public:
    String(const std::string& v) : m_val(v) {}

    String(char const* v) : m_val(v) {}

    static std::shared_ptr<String> make(const std::string& v) {
        return std::make_shared<String>(v);
    }

    bool operator==(const String& rhs) const { return m_val == rhs.m_val; }

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class Object final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    std::unordered_map<String, std::shared_ptr<Value>, StdHashAdaptor<String>> m_val;

public:
    static std::shared_ptr<Object> make() { return std::make_shared<Object>(); }

    static std::shared_ptr<Object> make(
            const std::vector<std::pair<String, std::shared_ptr<Value>>>& val) {
        for (auto&& i : val)
            mgb_assert(i.second);
        auto rst = make();
        rst->m_val.insert(val.begin(), val.end());
        return rst;
    }

    std::shared_ptr<Value>& operator[](const String& s) { return m_val[s]; }

    std::shared_ptr<Value>& operator[](const std::string& s) { return m_val[s]; }

    std::shared_ptr<Value>& operator[](const char* s) { return m_val[std::string(s)]; }

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class Array final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

    std::vector<std::shared_ptr<Value>> m_val;

public:
    static std::shared_ptr<Array> make() { return std::make_shared<Array>(); }

    void add(std::shared_ptr<Value> val) {
        mgb_assert(val);
        m_val.emplace_back(std::move(val));
    }

    std::shared_ptr<Value>& operator[](size_t idx) { return m_val.at(idx); }

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int indent = 0) const override;

    auto&& get_impl() { return m_val; }

    auto&& get_impl() const { return m_val; }
};

class Null final : public Value {
    MGB_DYN_TYPE_OBJ_FINAL_DECL_WITH_EXPORT;

public:
    static std::shared_ptr<Value> make() {
        static std::shared_ptr<Null> v(new Null);
        return v;
    }

    MGE_WIN_DECLSPEC_FUC void writeto(std::string& fout, int /*indent*/) const override;
};

class Serializable {
public:
    /*!
     * \brief dump internal state as json value
     */
    virtual std::shared_ptr<Value> to_json() const = 0;

    virtual ~Serializable() = default;
};

}  // namespace json

template <>
struct HashTrait<json::String> {
    static size_t eval(const json::String& s) { return hash(s.get_impl()); }
};

}  // namespace mgb

#else

namespace mgb {
namespace json {

class Serializable {};

}  // namespace json
}  // namespace mgb

#endif  // MGB_ENABLE_JSON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
