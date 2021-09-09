/**
 * \file src/custom/impl/param.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/custom/param.h"
#include "megbrain/common.h"
#include "megbrain/utils/hash.h"
#include <limits>
#include <sstream>
#include <map>

using namespace mgb;

namespace custom {

class ParamSchemaImpl {
    std::string m_name;
    std::string m_desc;
    ParamVal m_default;
    friend ParamSchema;
};

class ParamInfoImpl {
    std::vector<ParamSchema> m_meta;
    uint32_t TAG;
    friend ParamInfo;
};

class ParamImpl {
    std::unordered_map<std::string, ParamVal> m_vals;

    ParamImpl() = default;
    ParamImpl(const ParamImpl &rhs) = default;
    ParamImpl &operator=(const ParamImpl &rhs) {
        mgb_assert(
            m_vals.size() == rhs.m_vals.size(),
            "params of different op, assignment failed!"
        );
        for (const auto &kv: rhs.m_vals) {
            auto iter = m_vals.find(kv.first);
            mgb_assert(iter != m_vals.end(), "params of different op, assignment failed!");
            iter->second = kv.second;
        }
        return *this;
    }

    friend Param;
};

CUSTOM_PIMPL_CLS_DEFINE(ParamSchema)

ParamSchema::ParamSchema(const std::string &name, const ParamVal &value, const std::string &desc)
        : m_impl(new ParamSchemaImpl(), impl_deleter<ParamSchemaImpl>) {
    TypedRef(ParamSchemaImpl, m_impl.get()).m_name = name;
    TypedRef(ParamSchemaImpl, m_impl.get()).m_default = value;
    TypedRef(ParamSchemaImpl, m_impl.get()).m_desc = desc;
}

const std::string &ParamSchema::name(void) const {
    return TypedRef(ParamSchemaImpl, m_impl.get()).m_name;
}

const std::string &ParamSchema::desc(void) const {
    return TypedRef(ParamSchemaImpl, m_impl.get()).m_desc;
}

const ParamVal &ParamSchema::default_val(void) const {
    return TypedRef(ParamSchemaImpl, m_impl.get()).m_default;
}

ParamDynType ParamSchema::type(void) const {
    return TypedRef(ParamSchemaImpl, m_impl.get()).m_default.type();
}

std::string ParamSchema::str(void) const {
    std::stringstream ss;
    ss << "name: " << TypedRef(ParamSchemaImpl, m_impl.get()).m_name 
       << "\ndesc: " << TypedRef(ParamSchemaImpl, m_impl.get()).m_desc
       << "\n" << TypedRef(ParamSchemaImpl, m_impl.get()).m_default.str();
    return ss.str();
}

CUSTOM_PIMPL_CLS_DEFINE(ParamInfo)

void ParamInfo::set_tag(const std::string &hash_str) {
    const char *ptr = hash_str.c_str();
    TypedRef(ParamInfoImpl, m_impl.get()).TAG = 0;
    for (size_t i=0; i<hash_str.size(); i++) {
        TypedRef(ParamInfoImpl, m_impl.get()).TAG = 
            mgb::hash_pair_combine(TypedRef(ParamInfoImpl, m_impl.get()).TAG, mgb::hash(*(ptr++))) %
            std::numeric_limits<uint32_t>::max();
    }
}

void ParamInfo::set_meta(const std::vector<ParamSchema> &meta) {
    TypedRef(ParamInfoImpl, m_impl.get()).m_meta = meta;
}

uint32_t ParamInfo::tag(void) const {
    return TypedRef(ParamInfoImpl, m_impl.get()).TAG;
}

std::vector<ParamSchema> &ParamInfo::meta(void) {
    return TypedRef(ParamInfoImpl, m_impl.get()).m_meta;
}

const std::vector<ParamSchema> &ParamInfo::meta(void) const {
    return TypedRef(ParamInfoImpl, m_impl.get()).m_meta;
}

CUSTOM_PIMPL_CLS_DEFINE(Param)

Param::Param(const ParamInfo &info): m_impl(new ParamImpl(), impl_deleter<ParamImpl>) {
    for (const auto &schema: info.meta()) {
        TypedRef(ParamImpl, m_impl.get()).m_vals.emplace(schema.name(), schema.default_val());
    }
}

ParamVal &Param::operator[](const std::string &name) {
    return TypedRef(ParamImpl, m_impl.get()).m_vals.find(name)->second;
}

const ParamVal &Param::operator[](const std::string &name) const {
    return TypedRef(ParamImpl, m_impl.get()).m_vals.find(name)->second;
}

const std::unordered_map<std::string, ParamVal> &Param::raw() const {
    return TypedRef(ParamImpl, m_impl.get()).m_vals;
}

bool Param::exist(const std::string &name) const {
    return TypedRef(ParamImpl, m_impl.get()).m_vals.find(name) != 
           TypedRef(ParamImpl, m_impl.get()).m_vals.end();
}

std::string Param::to_bytes(void) const {
    std::string res;
    std::map<std::string, ParamVal> ordered_vals(
        TypedRef(ParamImpl, m_impl.get()).m_vals.begin(),
        TypedRef(ParamImpl, m_impl.get()).m_vals.end());
    for (auto &&kv: ordered_vals) {
        res += ParamVal::to_bytes(kv.second);
    }
    return res;
}

void Param::from_bytes(const std::string &bytes) {
    std::map<std::string, ParamVal> ordered_vals(
        TypedRef(ParamImpl, m_impl.get()).m_vals.begin(), 
        TypedRef(ParamImpl, m_impl.get()).m_vals.end());
    size_t offset = 0;
    for (auto &kv: ordered_vals) {
        kv.second = ParamVal::from_bytes(bytes, offset);
    }
    TypedRef(ParamImpl, m_impl.get()).m_vals.clear();
    TypedRef(ParamImpl, m_impl.get()).m_vals.insert(ordered_vals.begin(), ordered_vals.end());
    mgb_assert(offset == bytes.size(), "wrong data loader");
}

bool operator==(const Param &lhs, const Param &rhs) {
    if (lhs.raw().size() != rhs.raw().size())
        return false;
    for (const auto &kv: lhs.raw()) {
        auto riter = rhs.raw().find(kv.first);
        if (riter == rhs.raw().end() || !((kv.second) == riter->second)) {
            return false;
        }
    }
    return true;
}

}
