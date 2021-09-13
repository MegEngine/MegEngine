/**
 * \file src/custom/impl/manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include "megbrain/custom/manager.h"
#include <unordered_set>

#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace mgb;

namespace custom {

CustomOpManager *CustomOpManager::inst(void) {
    static CustomOpManager op_manager;
    return &op_manager;
}

CustomOpManager::~CustomOpManager() {
    mgb_assert(m_name2op.size() == m_id2op.size(), "Custom Op maintenance error!");
    LibManager::inst()->m_custom_libs.clear();
}

std::shared_ptr<CustomOp> CustomOpManager::insert(const std::string &name, uint32_t version) {
    MGB_LOCK_GUARD(m_mtx);
    auto iter = m_name2op.find(name);
    if (iter != m_name2op.end()) {
        mgb_log_warn("Register Custom Op Failed! Op %s has been registered", name.c_str());
        return std::const_pointer_cast<CustomOp, const CustomOp>(iter->second);
    }
    std::shared_ptr<const CustomOp> op = std::make_shared<const CustomOp>(name, version);
    m_name2op[op->op_type()] = op;
    m_id2op[op->runtime_id()] = op;
    return std::const_pointer_cast<CustomOp, const CustomOp>(op);
}

bool CustomOpManager::erase(const std::string &name) {
    MGB_LOCK_GUARD(m_mtx);
    auto iter = m_name2op.find(name);
    if (iter == m_name2op.end()) {
        mgb_log_warn("Erase Custom Op Failed! %s has not been registered", name.c_str());
        return false;
    }
    std::shared_ptr<const CustomOp> op = iter->second;
    m_id2op.erase(op->runtime_id());
    m_name2op.erase(op->op_type());
    return true;
}

bool CustomOpManager::erase(const RunTimeId &id) {
    MGB_LOCK_GUARD(m_mtx);
    auto iter = m_id2op.find(id);
    if (iter == m_id2op.end()) {
        mgb_log_warn("Erase Custom Op Failed! The Op has not been registered");
        return false;
    }
    std::shared_ptr<const CustomOp> op = iter->second;
    m_id2op.erase(op->runtime_id());
    m_name2op.erase(op->op_type());
    return true;
}

std::shared_ptr<CustomOp> CustomOpManager::find_or_reg(const std::string &name, uint32_t version) {
    auto iter = m_name2op.find(name);
    if (iter == m_name2op.end()) {
        return insert(name, version);
    }
    return std::const_pointer_cast<CustomOp, const CustomOp>(iter->second);
}

RunTimeId CustomOpManager::to_id(const std::string &name) const {
    std::shared_ptr<const CustomOp> op = find(name);
    return op->runtime_id();
}

std::string CustomOpManager::to_name(const RunTimeId &id) const {
    std::shared_ptr<const CustomOp> op = find(id);
    return op->op_type();
}

std::shared_ptr<const CustomOp> CustomOpManager::find(const std::string &name) const {
    auto ret = m_name2op.find(name);
    mgb_assert(ret != m_name2op.end(), 
        "Find Custom Op Failed! Op %s has not been registered", name.c_str()
    );
    return ret->second;
}

std::shared_ptr<const CustomOp> CustomOpManager::find(const RunTimeId &id) const {
    auto ret = m_id2op.find(id);
    mgb_assert(ret != m_id2op.end(), "Find Custom Op Failed! Op has not been registered");
    return ret->second;
}

std::vector<std::string> CustomOpManager::op_name_list(void) {
    std::vector<std::string> ret;
    for (auto kv: m_name2op) {
        ret.emplace_back(kv.first);
    }
    return ret;
}

std::vector<RunTimeId> CustomOpManager::op_id_list(void) {
    std::vector<RunTimeId> ret;
    for (auto kv: m_id2op) {
        ret.emplace_back(kv.first);
    }
    return ret;
}

#ifndef _WIN32
CustomLib::CustomLib(const std::string &path, int mode = RTLD_LAZY)
        : m_handle(nullptr, [](void* handle) {dlclose(handle);}) {
    auto op_list_before_load = CustomOpManager::inst()->op_name_list();
    std::unordered_set<std::string> op_set_before_load(
        op_list_before_load.begin(), op_list_before_load.end());

    m_handle.reset(dlopen(path.c_str(), mode));
    mgb_assert(m_handle != nullptr, "open custom op lib failed, error type: %s", dlerror());

    auto op_list_after_load = CustomOpManager::inst()->op_name_list();
    for (auto &op: op_list_after_load) {
        if (op_set_before_load.find(op) == op_set_before_load.end()) {
            m_ops.emplace_back(op);
        }
    }
}
#else
CustomLib::CustomLib(const std::string &path, int mode = 0)
        : m_handle(nullptr, [](void* handle) {}) {
    mgb_assert(false, "custom op is only supported on Linux now");
}
#endif

const std::vector<std::string> &CustomLib::ops_in_lib(void) const {
    return m_ops;
}

CustomLib::~CustomLib() {
    for (auto &op: m_ops) {
        CustomOpManager::inst()->erase(op);
    }
}

bool CustomLib::valid() const {
    return m_handle != nullptr;
}

LibManager *LibManager::inst(void) {
    static LibManager custom_libs;
    return &custom_libs;
}

const std::vector<std::string> &LibManager::install(const std::string &name, const std::string &path) {
    MGB_LOCK_GUARD(m_mtx);;
    LibHandle handle = std::make_shared<CustomLib>(path);
    m_custom_libs.insert({name, handle});
    return m_custom_libs[name]->ops_in_lib();
}

bool LibManager::uninstall(const std::string &name) {
    MGB_LOCK_GUARD(m_mtx);;
    mgb_assert(m_custom_libs.erase(name) == 1, "uninstall error");
    return true;
}

std::shared_ptr<CustomOp> op_insert(std::string opname, uint32_t version) {
    return CustomOpManager::inst()->insert(opname, version);
}

}

#endif
