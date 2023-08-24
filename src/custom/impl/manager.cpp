#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include <unordered_set>
#include "megbrain/custom/manager.h"

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#endif

using namespace mgb;

namespace custom {

#ifdef _WIN32
#define RTLD_LAZY 0

void* dlopen(const char* file, int) {
    return static_cast<void*>(LoadLibrary(file));
}

int dlclose(void* handle) {
    return static_cast<int>(FreeLibrary(static_cast<HMODULE>(handle)));
}

const char* dlerror(void) {
    static char win_err_info[] = "no dlerror info in windows";
    return win_err_info;
}
#endif

CustomLib::CustomLib(const std::string& path, int mode = RTLD_LAZY)
        : m_handle(nullptr, [](void* handle) { dlclose(handle); }) {
    auto op_list_before_load = CustomOpManager::inst()->op_name_list();
    std::unordered_set<std::string> op_set_before_load(
            op_list_before_load.begin(), op_list_before_load.end());

    m_handle.reset(dlopen(path.c_str(), mode));
    mgb_assert(
            m_handle != nullptr, "open custom op lib failed, error type: %s",
            dlerror());

    auto op_list_after_load = CustomOpManager::inst()->op_name_list();
    for (auto& op : op_list_after_load) {
        if (op_set_before_load.find(op) == op_set_before_load.end()) {
            m_ops.emplace_back(op);
        }
    }
}

CustomLib::~CustomLib() {
    for (auto& op : m_ops) {
        CustomOpManager::inst()->erase(op);
    }
}

const std::vector<std::string>& CustomLib::ops_in_lib(void) const {
    return m_ops;
}

bool CustomLib::valid() const {
    return m_handle != nullptr;
}

CustomOpManager* CustomOpManager::inst(void) {
    static CustomOpManager op_manager;
    return &op_manager;
}

CustomOpManager::~CustomOpManager() {
    mgb_assert(m_name2op.size() == m_id2op.size(), "Custom Op maintenance error!");
    {
        MGB_LOCK_GUARD(m_lib_mtx);
        m_custom_libs.clear();
    }

    mgb_assert(m_name2op.size() == m_id2op.size(), "Custom Op maintenance error!");
    MGB_LOCK_GUARD(m_op_mtx);
    m_name2op.clear();
    m_id2op.clear();
}

const std::vector<std::string>& CustomOpManager::install(
        const std::string& name, const std::string& path) {
    MGB_LOCK_GUARD(m_lib_mtx);
    LibHandle handle = std::make_shared<CustomLib>(path);
    m_custom_libs.insert({name, handle});
    return m_custom_libs[name]->ops_in_lib();
}

std::vector<std::string> CustomOpManager::uninstall(const std::string& name) {
    MGB_LOCK_GUARD(m_lib_mtx);
    std::vector<std::string> op_names = m_custom_libs[name]->ops_in_lib();
    mgb_assert(m_custom_libs.erase(name) == 1, "uninstall error");
    return op_names;
}

const std::unordered_map<std::string, LibHandle>& CustomOpManager::lib_info(
        void) const {
    return m_custom_libs;
}

std::shared_ptr<CustomOp> CustomOpManager::insert(
        const std::string& name, uint32_t version) {
    MGB_LOCK_GUARD(m_op_mtx);
    auto iter = m_name2op.find(name);
    if (iter != m_name2op.end()) {
        mgb_log_warn(
                "Register Custom Op Failed! Op %s has been registered", name.c_str());
        return std::const_pointer_cast<CustomOp, const CustomOp>(iter->second);
    }
    std::shared_ptr<const CustomOp> op =
            std::make_shared<const CustomOp>(name, version);
    m_name2op[op->op_type()] = op;
    m_id2op[op->runtime_id()] = op;
    return std::const_pointer_cast<CustomOp, const CustomOp>(op);
}

bool CustomOpManager::erase(const std::string& name) {
    MGB_LOCK_GUARD(m_op_mtx);
    auto iter = m_name2op.find(name);
    if (iter == m_name2op.end()) {
        mgb_log_warn(
                "Erase Custom Op Failed! %s has not been registered", name.c_str());
        return false;
    }
    std::shared_ptr<const CustomOp> op = iter->second;
    m_id2op.erase(op->runtime_id());
    m_name2op.erase(op->op_type());
    return true;
}

RunTimeId CustomOpManager::to_id(const std::string& name) const {
    std::shared_ptr<const CustomOp> op = find(name);
    return op->runtime_id();
}

std::string CustomOpManager::to_name(const RunTimeId& id) const {
    std::shared_ptr<const CustomOp> op = find(id);
    return op->op_type();
}

std::shared_ptr<const CustomOp> CustomOpManager::find(const std::string& name) const {
    auto ret = m_name2op.find(name);
    mgb_assert(
            ret != m_name2op.end(),
            "Find Custom Op Failed! Op %s has not been registered", name.c_str());
    return ret->second;
}

std::shared_ptr<const CustomOp> CustomOpManager::find(const RunTimeId& id) const {
    auto ret = m_id2op.find(id);
    mgb_assert(
            ret != m_id2op.end(), "Find Custom Op Failed! Op has not been registered");
    return ret->second;
}

std::vector<std::string> CustomOpManager::op_name_list(void) {
    std::vector<std::string> ret;
    for (auto kv : m_name2op) {
        ret.emplace_back(kv.first);
    }
    return ret;
}

std::vector<RunTimeId> CustomOpManager::op_id_list(void) {
    std::vector<RunTimeId> ret;
    for (auto kv : m_id2op) {
        ret.emplace_back(kv.first);
    }
    return ret;
}

std::shared_ptr<CustomOp> op_insert(std::string opname, uint32_t version) {
    return CustomOpManager::inst()->insert(opname, version);
}

}  // namespace custom

#endif
