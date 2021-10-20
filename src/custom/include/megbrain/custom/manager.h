/**
 * \file src/custom/include/megbrain/custom/manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
    static CustomOpManager* inst(void);
    ~CustomOpManager();

    std::shared_ptr<CustomOp> insert(const std::string& name, uint32_t version);
    bool erase(const std::string& name);
    bool erase(const RunTimeId& id);

    std::shared_ptr<CustomOp> find_or_reg(const std::string& name, uint32_t version);

    RunTimeId to_id(const std::string& name) const;
    std::string to_name(const RunTimeId& id) const;

    std::shared_ptr<const CustomOp> find(const std::string& name) const;
    std::shared_ptr<const CustomOp> find(const RunTimeId& id) const;

    std::vector<std::string> op_name_list(void);
    std::vector<RunTimeId> op_id_list(void);
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

    static LibManager* inst(void);
    const std::vector<std::string>& install(
            const std::string& name, const std::string& path);
    bool uninstall(const std::string& name);
    friend class CustomOpManager;
};

}  // namespace custom
