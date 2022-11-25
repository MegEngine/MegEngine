#include "megbrain/imperative/value.h"

#include "megbrain/imperative/basic_operators.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/utils/map.h"

namespace mgb {
namespace imperative {

namespace {
static /*thread_local*/ size_t nr_watched_values = 0;
static /*thread_local*/ uint64_t nr_values = 0;
static /*thread_local*/ bool recording_values = false;
static /*thread_local*/ std::vector<ValueWeakRef> recorded_values;
static WeakValueMap<uint64_t, ValueWeakRef> registered_values;
}  // namespace

ValueRef::storage_t& ValueRef::storage() const {
    if (mgb_likely(!m_storage->m_successor.m_storage)) {
        return m_storage;
    }
    while (m_storage->m_successor.m_storage) {
        m_storage = m_storage->m_successor.m_storage;
    }
    return m_storage;
}

const Value* ValueRef::as(const IType& type) const {
    auto&& storage = this->storage();
    if (storage->type() != type) {
        return nullptr;
    }
    return static_cast<Value*>(storage.get());
}

bool ValueRef::is(const IType& type) const {
    return this->storage()->type() == type;
}

TypedValueRef<DeviceValue> ValueRef::dev_tensor() const {
    return imperative::apply(GetAttr(GetAttr::Data), *this)[0].cast_ref<DeviceValue>();
}

TypedValueRef<HostValue> ValueRef::numpy() const {
    return imperative::apply(GetAttr(GetAttr::Value), *this)[0].cast_ref<HostValue>();
}

TypedValueRef<CompNodeValue> ValueRef::device() const {
    return imperative::apply(GetAttr(GetAttr::Device), *this)[0]
            .cast_ref<CompNodeValue>();
}

TypedValueRef<ShapeValue> ValueRef::shape() const {
    return imperative::apply(GetAttr(GetAttr::Shape), *this)[0].cast_ref<ShapeValue>();
}

TypedValueRef<DTypeValue> ValueRef::dtype() const {
    return imperative::apply(GetAttr(GetAttr::DType), *this)[0].cast_ref<DTypeValue>();
}

TypedValueRef<FormatValue> ValueRef::format() const {
    return imperative::apply(GetFormat(), *this)[0].as_ref<FormatValue>();
}

TypedValueRef<StringValue> ValueRef::name() const {
    return imperative::apply(GetName(), *this)[0].cast_ref<StringValue>();
}

bool ValueRef::is_scalar() const {
    return imperative::apply(IsScalar(), *this)[0].cast<BoolValue>();
}

void ValueRef::watch() const {
    mgb_assert(m_storage);
    storage()->m_watching++;
    nr_watched_values++;
    storage()->on_watch();
    // TODO:
    // imperative::apply(Watch(), this);
}

void ValueRef::unwatch() const {
    mgb_assert(m_storage);
    storage()->m_watching--;
    nr_watched_values--;
    storage()->on_unwatch();
}

ValueRef ValueRef::unwrap() const {
    auto& context = Transformation::get_context();
    if (mgb_unlikely(context.next_transformation)) {
        ValueRef value = *this;
        for (size_t i = 0; i < context.next_transformation; ++i) {
            value = context.transformations[i]->unwrap(value);
        }
        return value;
    }
    return *this;
}

std::string ValueRef::to_string() const {
    if (!m_storage) {
        return "<empty value>";
    }
    return ssprintf(
            "(%zu:%zu) %s", id(), storage()->m_id, storage()->to_string().c_str());
}

std::string ValueRef::raw_type() const {
    if (!m_storage) {
        return "null";
    }
    return this->storage()->type().name();
}

const IType* ValueRef::type() const {
    if (!m_storage) {
        return nullptr;
    }
    return &m_storage->type();
}

bool ValueRef::watching() const {
    if (!m_storage) {
        return false;
    }
    return this->storage()->m_watching;
}

ValueRef ValueRef::make(ValueRef::storage_t storage) {
    if (recording_values) {
        recorded_values.push_back({storage});
    }
    return {storage};
}

bool ValueRef::any_watching() {
    return nr_watched_values != 0;
}

ValueRef ValueWeakRef::lock() {
    auto strong_storage = m_storage.lock();
    if ((!strong_storage) || strong_storage->m_successor) {
        return {};
    }
    return {strong_storage};
}

Value::Value() {
    m_id = nr_values++;
}

Value::~Value() {
    if (m_watching) {
        debug::notify_event("dtor");
    }
}

void Value::register_value(ValueRef value) {
    registered_values[value.id()] = ValueWeakRef(value);
}

ValueRef Value::get_value_by_id(uint64_t id) {
    auto& weak_value = registered_values[id];
    if (auto value = weak_value.lock()) {
        return value;
    }
    return {};
}

void Value::begin_record_values() {
    mgb_assert(!recording_values);
    recording_values = true;
    recorded_values.clear();
}

std::vector<ValueRef> Value::end_record_values() {
    recording_values = false;
    std::vector<ValueRef> recorded_strong_values;
    for (auto&& weak_value : recorded_values) {
        if (auto value = weak_value.lock()) {
            recorded_strong_values.push_back(value);
        }
    }
    return recorded_strong_values;
}

void Value::try_rethrow() {
    if (type() == PrimitiveType<ErrorValue>::instance) {
        auto message = static_cast<ErrorValue*>(this)->message();
        mgb_throw(MegBrainError, "invalid value: %s", message.c_str());
    }
}

inline void ValueRefList::init(size_t nr_elems) {
    m_size = nr_elems;
    if (m_size > 0) {
        if (m_size == 1) {
            m_data = new (inline_storage()) ValueRef();
        } else {
            m_data = new ValueRef[m_size];
        }
    } else {
        m_data = nullptr;
    }
}

ValueRefList::ValueRefList(size_t nr_elems) {
    init(nr_elems);
}

ValueRefList::ValueRefList(const ValueRefList& rhs)
        : ValueRefList(rhs.cbegin(), rhs.cend()) {}

ValueRefList::ValueRefList(ValueRefList&& rhs) : ValueRefList() {
    m_size = rhs.m_size;
    if (rhs.m_data == rhs.inline_storage()) {
        m_data = inline_storage();
        new (m_data) ValueRef();
        m_data[0] = std::move(rhs.m_data[0]);
    } else {
        m_data = rhs.m_data;
        rhs.m_data = nullptr;
        rhs.m_size = 0;
    }
}

ValueRefList& ValueRefList::operator=(const ValueRefList& rhs) {
    if (this == &rhs) {
        return *this;
    }
    clear();
    init(rhs.m_size);
    for (size_t i = 0; i < m_size; ++i) {
        m_data[i] = rhs.m_data[i];
    }
    return *this;
}

ValueRefList& ValueRefList::operator=(ValueRefList&& rhs) {
    if (this == &rhs) {
        return *this;
    }
    clear();
    if (rhs.m_data == rhs.inline_storage()) {
        m_data = inline_storage();
        new (m_data) ValueRef();
        m_data[0] = rhs.m_data[0];
        m_size = 1;
        rhs.clear();
    } else {
        m_data = rhs.m_data;
        m_size = rhs.m_size;
        rhs.m_data = nullptr;
        rhs.m_size = 0;
    }
    return *this;
}

ValueRefList::~ValueRefList() {
    clear();
}

void ValueRefList::clear() {
    if (m_data) {
        if (m_size != 1) {
            delete[] m_data;
        } else {
            mgb_assert(m_data == inline_storage());
            m_data->~ValueRef();
        }
    }
    m_data = nullptr;
    m_size = 0;
}

}  // namespace imperative
}  // namespace mgb
