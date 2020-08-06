/**
 * \file src/serialization/impl/opr_registry.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/serialization/opr_registry.h"

#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/utils/hash_ct.h"

using namespace mgb;
using namespace serialization;

namespace mgb {
    //! implemented in sereg_caller.cpp, so this file can depend on call_sereg
    //! and the registries would not be stripped when linked statically
    void call_sereg();
}

namespace {
    struct StaticData {
        //! all oprs must have ID; but legacy oprs only have ID without Typeinfo
        ThinHashMap<size_t, OprRegistry> id2reg;
        ThinHashMap<Typeinfo*, OprRegistry*> type2reg;
        std::unordered_map<std::string, OprRegistry*> name2reg;
        ThinHashMap<size_t, OprRegistry*> unversioned_id2reg;
    };

    StaticData& static_data() {
        // to ensure static data can be initialized before calling add()
        static StaticData inst;
        return inst;
    }

    cg::OperatorNodeBase* dynamic_loader(
            OprLoadContext &ctx, const cg::VarNodeArray &inputs,
            const OperatorNodeConfig &config) {
        auto name = ctx.load_buf_with_len();
        return ctx.make_opr_loader(name)(ctx, inputs, config);
    }

    const OprRegistry* dynamic_registry() {
        static const OprRegistry* ret = nullptr;
        if (ret)
            return ret;

        auto id = MGB_HASH_STR("dynamic");
        OprRegistry::add({nullptr, id, {}, {}, dynamic_loader, {}, id});
        ret = OprRegistry::find_by_id(id);
        mgb_assert(ret);
        return ret;
    }

    class _Init {
        public:
            _Init() {
                call_sereg();
                dynamic_registry();
            }
    };
    _Init _init;
} // anonymous namespace


void OprRegistry::add(const OprRegistry& record) {
    auto&& sd = static_data();

    auto persist_id = record.persist_type_id;
    auto registry_ins = sd.id2reg.emplace(persist_id, record);
    mgb_assert(registry_ins.second ||
                       persist_id == dynamic_registry()->persist_type_id,
               "duplicated operator persist_type_id: %s",
               std::to_string(persist_id).c_str());

    OprRegistry* persis_record_ptr;
    if (registry_ins.second) {
        persis_record_ptr = &registry_ins.first->second;
    } else {
        static std::vector<std::unique_ptr<OprRegistry>> dynamic_opr_reg;
        mgb_assert(!record.loader);
        dynamic_opr_reg.emplace_back(new OprRegistry{record});
        persis_record_ptr = dynamic_opr_reg.back().get();
    }

    if (!record.type) {
        // loader only for compatibility
        mgb_assert(!record.dumper);
        mgb_assert(!record.shallow_copy);
    } else {
        auto&& res = sd.type2reg.insert({record.type, persis_record_ptr});
        mgb_assert(res.second, "duplicated OprRegistry type: %s",
                record.type->name);
        if (!record.shallow_copy) {
            res.first->second->shallow_copy =
                    intl::copy_opr_shallow_default_impl;
        }
    }

    if (!record.name.empty()) {
        auto&& n2r = sd.name2reg[record.name];
        mgb_assert(!n2r, "duplicated OprRegistry name: %s",
                   record.name.c_str());
        n2r = persis_record_ptr;
    }

    if (record.unversioned_type_id) {
        auto&& res = sd.unversioned_id2reg.emplace(record.unversioned_type_id,
                                                   persis_record_ptr);
        mgb_assert(
                res.second || record.unversioned_type_id ==
                                      dynamic_registry()->unversioned_type_id,
                "duplicated OprRegistry unversioned id: %s",
                std::to_string(record.unversioned_type_id).c_str());
    }
}

const OprRegistry* OprRegistry::find_by_name(const std::string &name) {
    auto &&name2reg = static_data().name2reg;
    auto iter = name2reg.find(name);
    return iter == name2reg.end() ? nullptr : iter->second;
}

const OprRegistry* OprRegistry::find_by_id(size_t id) {
    auto &&id2reg = static_data().id2reg;
    auto iter = id2reg.find(id);
    return iter == id2reg.end() ? nullptr : &iter->second;
}

const OprRegistry* OprRegistry::find_by_type(Typeinfo* type) {
    auto &&type2reg = static_data().type2reg;
    auto iter = type2reg.find(type);
    return iter == type2reg.end() ? nullptr : iter->second;
}

const OprRegistry* OprRegistry::find_by_unversioned_id(size_t unversioned_id) {
    auto &&uid2reg = static_data().unversioned_id2reg;
    auto iter = uid2reg.find(unversioned_id);
    return iter == uid2reg.end() ? nullptr : iter->second;
}

void OprRegistry::add_using_dynamic_loader(
        Typeinfo *type, const std::string &name, const OprDumper &dumper) {
    // dynamic oprs are implemented by mapping different opr types to the same
    // persist_type_id
    add({type,
         dynamic_registry()->persist_type_id,
         name,
         dumper,
         {},
         {},
         dynamic_registry()->unversioned_type_id});
}

#if MGB_ENABLE_DEBUG_UTIL
std::vector<std::pair<size_t, std::string>> OprRegistry::dump_registries() {
    auto&& id2reg = static_data().id2reg;
    std::vector<std::pair<size_t, std::string>> result;
    for (auto iter = id2reg.begin(); iter != id2reg.end(); ++iter) {
        if (iter->second.name.size() == 0)
            result.push_back({iter->first, "<special>"});
        else
            result.push_back({iter->first, iter->second.name});
    }
    return result;
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
