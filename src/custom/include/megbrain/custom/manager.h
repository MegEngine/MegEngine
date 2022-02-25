#pragma once

#include "custom.h"
#include "megbrain/common.h"

namespace custom {

class CustomOpManager {
    std::unordered_map<std::string, std::shared_ptr<const CustomOp>> m_name2op;
    std::unordered_map<RunTimeId, std::shared_ptr<const CustomOp>> m_id2op;
    MGB_MUTEX m_mtx;
    CustomOpManager() = default;

public:
    PREVENT_COPY_AND_ASSIGN(CustomOpManager);
    MGE_WIN_DECLSPEC_FUC static CustomOpManager* inst(void);
    MGE_WIN_DECLSPEC_FUC ~CustomOpManager();

    MGE_WIN_DECLSPEC_FUC std::shared_ptr<CustomOp> insert(
            const std::string& name, uint32_t version);
    MGE_WIN_DECLSPEC_FUC bool erase(const std::string& name);
    MGE_WIN_DECLSPEC_FUC bool erase(const RunTimeId& id);

    MGE_WIN_DECLSPEC_FUC std::shared_ptr<CustomOp> find_or_reg(
            const std::string& name, uint32_t version);

    MGE_WIN_DECLSPEC_FUC RunTimeId to_id(const std::string& name) const;
    MGE_WIN_DECLSPEC_FUC std::string to_name(const RunTimeId& id) const;

    MGE_WIN_DECLSPEC_FUC std::shared_ptr<const CustomOp> find(
            const std::string& name) const;
    MGE_WIN_DECLSPEC_FUC std::shared_ptr<const CustomOp> find(
            const RunTimeId& id) const;

    MGE_WIN_DECLSPEC_FUC std::vector<std::string> op_name_list(void);
    MGE_WIN_DECLSPEC_FUC std::vector<RunTimeId> op_id_list(void);
};

class CustomLib {
    std::unique_ptr<void, void_deleter> m_handle;
    std::vector<std::string> m_ops;

public:
    PREVENT_COPY_AND_ASSIGN(CustomLib);

    CustomLib(const std::string& path, int mode);
    const std::vector<std::string>& ops_in_lib(void) const;
    ~CustomLib();
    bool valid(void) const;
};

using LibHandle = std::shared_ptr<CustomLib>;

class LibManager {
    std::unordered_map<std::string, LibHandle> m_custom_libs;
    MGB_MUTEX m_mtx;

    LibManager() = default;

public:
    PREVENT_COPY_AND_ASSIGN(LibManager);

    MGE_WIN_DECLSPEC_FUC static LibManager* inst(void);
    MGE_WIN_DECLSPEC_FUC const std::vector<std::string>& install(
            const std::string& name, const std::string& path);
    MGE_WIN_DECLSPEC_FUC bool uninstall(const std::string& name);
    friend class CustomOpManager;
};

}  // namespace custom
