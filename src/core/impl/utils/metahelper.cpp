/**
 * \file src/core/impl/utils/metahelper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/metahelper.h"

using namespace mgb;

class None mgb::None;

void metahelper_detail::on_maybe_invalid_val_access() {
    mgb_throw(InternalError, "access invalid Maybe value");
}

UserDataContainer::~UserDataContainer() noexcept = default;

void UserDataContainer::clear_all_user_data() {
    m_refkeeper.clear();
    m_storage.clear();
}

void UserDataContainer::do_add(Typeinfo *type, std::shared_ptr<UserData> data) {
    auto ins = m_refkeeper.emplace(std::move(data));
    mgb_assert(ins.second, "duplicated user data: %p", ins.first->get());
    m_storage[type].push_back(ins.first->get());
}

std::pair<void*const*, size_t> UserDataContainer::do_get(Typeinfo *type) const {
    auto iter = m_storage.find(type);
    if (iter == m_storage.end())
        return {nullptr, 0};
    auto &&vec = iter->second;
    return {vec.data(), vec.size()};
}

void* UserDataContainer::do_get_one(Typeinfo *type) const {
    auto &&vec = m_storage.at(type);
    return vec.back();
}

int UserDataContainer::do_pop(Typeinfo *type) {
    auto iter = m_storage.find(type);
    if (iter == m_storage.end())
        return 0;
    auto &&vec = iter->second;
    mgb_assert(!vec.empty());
    // use aliasing constructor to avoid deleter call
    std::shared_ptr<UserData> ptr(std::shared_ptr<UserData>{},
            static_cast<UserData*>(vec.back()));
    auto nr = m_refkeeper.erase(ptr);
    mgb_assert(nr);
    vec.pop_back();
    if (vec.empty()) {
        m_storage.erase(iter);
    }
    return 1;
}

void CleanupCallback::add(Callback callback) {
    m_callbacks.emplace_back(std::move(callback));
}

CleanupCallback::~CleanupCallback() noexcept(false) {
    for (auto&& i : reverse_adaptor(m_callbacks)) {
#if MGB_ENABLE_EXCEPTION
        std::exception_ptr exc = nullptr;
#endif
        MGB_TRY { i(); }
        MGB_CATCH_ALL_EXCEPTION("cleanup callback", exc);

#if MGB_ENABLE_EXCEPTION
        if (exc) {
            if (mgb::has_uncaught_exception()) {
                mgb_log_error(
                        "ignore exception from cleanup callbacks due to "
                        "uncaught exception");
            } else {
                std::rethrow_exception(exc);
            }
        }
#endif
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
