#pragma once

#include <list>
#include <map>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "megbrain/common.h"
#include "megbrain/imperative/utils/span.h"
#include "megbrain/imperative/value.h"

namespace mgb {
namespace imperative {

using GenericFunction = std::function<ValueRefList(Span<ValueRef>)>;

/**
 * \brief base class for all operators
 *
 */
class Operator {
public:
    enum Kind {
        IdentityLike,  // one input, one output, output is like input
        GetAttrLike,   // no tensor output
        Other,
    };

private:
    size_t m_typecode;
    Kind m_kind;

protected:
    Operator(size_t typecode, Kind kind) : m_typecode{typecode}, m_kind{kind} {}

public:
    size_t typecode() const { return m_typecode; }
    Kind kind() const { return m_kind; }

    template <typename U>
    const U* as() const {
        if (m_typecode != U::TYPE_CODE) {
            return nullptr;
        }
        return static_cast<const U*>(this);
    }
    template <typename U>
    bool is() const {
        return m_typecode == U::TYPE_CODE;
    }
    template <Kind kKind>
    bool is() const {
        return kind() == kKind;
    }
    template <typename U>
    const U& cast() const {
        mgb_assert(m_typecode == U::TYPE_CODE);
        return static_cast<const U&>(*this);
    }

    virtual std::string to_string() const = 0;

    /**
     * \brief fallback implementation of this. Not all operators has fallback
     * implementation.
     *
     * \param inputs
     * \return ValueRefList
     */
    virtual ValueRefList fallback(Span<ValueRef> inputs) const;

    std::type_index type() const { return registered_types()[m_typecode]; }

    static size_t register_type(std::type_index type);
    static const std::vector<std::type_index>& registered_types();
};

template <typename T, Operator::Kind kKind = Operator::Other>
class OperatorImpl : public Operator {
protected:
    OperatorImpl() : Operator(TYPE_CODE, kKind) {}

public:
    static inline size_t TYPE_CODE = [] { return register_type(typeid(T)); }();

    std::string to_string() const override = 0;
};

}  // namespace imperative
}  // namespace mgb
