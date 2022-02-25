#pragma once

#include <optional>

#include "megbrain/utils/metahelper.h"

namespace mgb::imperative {

/**
 * \brief an hash map optimized for weak pointer as key
 *
 * Keys were scanned automatically, so values referenced by invalid keys whould be
 * released soon
 *
 * \tparam TKey key type, requires(bool(key.lock()))
 * \tparam TValue value type
 */
template <typename TKey, typename TValue>
class WeakKeyMap : public NonCopyableObj {
public:
    using storage_t = std::unordered_map<TKey, TValue>;

private:
    storage_t m_storage;
    typename storage_t::iterator m_cursor = m_storage.begin();

    /**
     * \brief select a key and verify that whether it is invalid. If yes, erase it
     *
     */
    void _step() {
        if (m_cursor == m_storage.end()) {
            m_cursor = m_storage.begin();
            return;
        }
        auto key = m_cursor->first;
        if (!key.lock()) {
            m_cursor = m_storage.erase(m_cursor);
        } else {
            ++m_cursor;
        }
    }

public:
    size_t count(TKey key) {
        _step();
        _step();
        return m_storage.count(key);
    }

    TValue& at(TKey key) const { return m_storage.at(key); }

    TValue& at(TKey key) {
        _step();
        _step();
        return m_storage.at(key);
    }

    TValue& operator[](TKey key) {
        _step();
        _step();
        if (m_storage.count(key)) {
            return m_storage.at(key);
        } else {
            size_t bucket_count = m_storage.bucket_count();
            TValue& result = m_storage[key];
            if (bucket_count != m_storage.bucket_count()) {
                m_cursor = m_storage.begin();
            }
            return result;
        }
    }

    std::optional<TValue> try_get(TKey key) const {
        auto iter = m_storage.find(key);
        if (iter == m_storage.end()) {
            return {};
        }
        return {iter->second};
    }

    std::optional<TValue> try_get(TKey key) {
        _step();
        _step();
        return ((const WeakKeyMap*)this)->try_get(std::move(key));
    }
};

template <typename TKey, typename TValue>
class WeakValueMap : public NonCopyableObj {
public:
    using storage_t = std::unordered_map<TKey, TValue>;

private:
    storage_t m_storage;
    typename storage_t::iterator m_cursor = m_storage.begin();

    /**
     * \brief select a key and verify that whether it is invalid. If yes, erase it
     *
     */
    void _step() {
        if (m_cursor == m_storage.end()) {
            m_cursor = m_storage.begin();
            return;
        }
        auto value = m_cursor->second;
        if (!value.lock()) {
            m_cursor = m_storage.erase(m_cursor);
        } else {
            ++m_cursor;
        }
    }

public:
    size_t count(TKey key) {
        _step();
        _step();
        return m_storage.count(key);
    }

    TValue& at(TKey key) const { return m_storage.at(key); }

    TValue& at(TKey key) {
        _step();
        _step();
        return m_storage.at(key);
    }

    TValue& operator[](TKey key) {
        _step();
        _step();
        if (m_storage.count(key)) {
            return m_storage.at(key);
        } else {
            size_t bucket_count = m_storage.bucket_count();
            TValue& result = m_storage[key];
            if (bucket_count != m_storage.bucket_count()) {
                m_cursor = m_storage.begin();
            }
            return result;
        }
    }
};

}  // namespace mgb::imperative