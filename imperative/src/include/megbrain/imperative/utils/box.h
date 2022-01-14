/**
 * \file imperative/src/include/megbrain/imperative/utils/visit.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <chrono>
#include <future>
#include <vector>

#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/small_vector.h"

namespace mgb::imperative {

class BoxBase : public NonCopyableObj {
public:
    virtual void reset() = 0;
    virtual void set_exception(std::exception_ptr exc) = 0;
    virtual bool try_set_exception(std::exception_ptr exc) = 0;
};

/**
 * \brief An reusable promise
 *
 * \tparam T type of value
 */
template <typename T>
class Box final : public BoxBase {
private:
    std::promise<T> m_promise;
    std::shared_future<T> m_future;
    std::mutex m_mutex;
    bool m_value_set;
    bool m_exception_set;

public:
    Box() { reset(); }
    const T& get_value() { return m_future.get(); }
    T take_value() {
        T value = m_future.get();
        reset();
        return value;
    }
    void set_value(T value) {
        MGB_LOCK_GUARD(m_mutex);
        m_promise.set_value(std::move(value));
        m_value_set = true;
    }
    bool try_set_value(T value) {
        MGB_LOCK_GUARD(m_mutex);
        if (m_exception_set) {
            return false;
        }
        m_promise.set_value(std::move(value));
        m_value_set = true;
        return true;
    }
    void set_exception(std::exception_ptr exc) override {
        MGB_LOCK_GUARD(m_mutex);
        m_promise.set_exception(exc);
        m_exception_set = true;
    }
    bool try_set_exception(std::exception_ptr exc) override {
        MGB_LOCK_GUARD(m_mutex);
        if (m_value_set) {
            return false;
        }
        m_promise.set_exception(exc);
        m_exception_set = true;
        return true;
    }
    void reset() override {
        MGB_LOCK_GUARD(m_mutex);
        m_promise = {};
        m_future = m_promise.get_future();
        m_value_set = false;
        m_exception_set = false;
    }

    /**
     * \brief make an empty box
     *
     * \return std::shared_ptr<Box>
     */
    static std::shared_ptr<Box> make() { return std::make_shared<Box>(); }
};

}  // namespace mgb::imperative
