/**
 * \file src/custom/include/megbrain/custom/param.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include "param_val.h"

namespace custom {

class ParamSchemaImpl;
class ParamInfoImpl;
class ParamImpl;

// Schema of a param element
class ParamSchema {
    CUSTOM_PIMPL_CLS_DECL(ParamSchema);
    ParamSchema(const std::string &name, const ParamVal &value, const std::string &desc="");

    const std::string &name(void) const;
    const std::string &desc(void) const;
    const ParamVal &default_val(void) const;
    ParamDynType type(void) const;
    std::string str(void) const;
};

class ParamInfo {
    CUSTOM_PIMPL_CLS_DECL(ParamInfo);

    void set_tag(const std::string&);
    void set_meta(const std::vector<ParamSchema> &meta);
    uint32_t tag(void) const;
    std::vector<ParamSchema> &meta(void);
    const std::vector<ParamSchema> &meta(void) const;
};

class Param {
    CUSTOM_PIMPL_CLS_DECL(Param);
    
    Param(const ParamInfo&);
    ParamVal &operator[](const std::string&);
    const ParamVal &operator[](const std::string&) const;
    const std::unordered_map<std::string, ParamVal> &raw() const;
    bool exist(const std::string &name) const;
    std::string to_bytes(void) const;
    void from_bytes(const std::string&);
};

bool operator==(const Param&, const Param&);

}   // custom
