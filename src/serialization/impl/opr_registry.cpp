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
}  // namespace mgb

namespace {
struct StaticData {
    //! all oprs must have ID; but legacy oprs only have ID without Typeinfo
    ThinHashMap<size_t, OprRegistry> id2reg;
    ThinHashMap<Typeinfo*, OprRegistry*> type2reg;
    std::unordered_map<std::string, OprRegistry*> name2reg;
    ThinHashMap<size_t, OprRegistry*> unversioned_id2reg;

    //! versioned OprRegistryV2, version_id_reg_map is used for Operator
    //! load/shallow copy and version_type_reg_map is used for Operator dump
    ThinHashMap<uint8_t, ThinHashMap<size_t, OprRegistryV2>> version_id_reg_map;
    ThinHashMap<uint8_t, ThinHashMap<Typeinfo*, OprRegistryV2*>> version_type_reg_map;
#if MGB_ENABLE_DEBUG_UTIL
    std::unordered_map<size_t, std::unordered_map<size_t, std::string>> dumped_opr;
    MGB_MUTEX g_record_map_mtx;
    bool recorded = false;
#endif
};

StaticData& static_data() {
    // to ensure static data can be initialized before calling add()
    static StaticData inst;
    return inst;
}

OprWithOutputAccessor dynamic_loader(
        OprLoadContext& ctx, const cg::VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
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

const OprRegistryV2* dynamic_registry_v2() {
    static const OprRegistryV2* ret = nullptr;
    if (ret)
        return ret;

    auto id = MGB_HASH_STR("dynamic");
    OprRegistryV2::versioned_add(
            {nullptr, id, {}, {}, dynamic_loader, {}}, CURRENT_VERSION,
            CURRENT_VERSION);
    ret = OprRegistryV2::versioned_find_by_id(id, CURRENT_VERSION);
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
}  // anonymous namespace

void OprRegistry::add(const OprRegistry& record) {
    auto&& sd = static_data();
    auto persist_id = record.persist_type_id;
    auto registry_ins = sd.id2reg.emplace(persist_id, record);
    mgb_assert(
            registry_ins.second || persist_id == dynamic_registry()->persist_type_id,
            "duplicated operator name : %s", record.name.c_str());

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
        mgb_assert(res.second, "duplicated OprRegistry type: %s", record.type->name);
        if (!record.shallow_copy) {
            res.first->second->shallow_copy = intl::copy_opr_shallow_default_impl;
        }
    }

    if (!record.name.empty()) {
        auto&& n2r = sd.name2reg[record.name];
        mgb_assert(!n2r, "duplicated OprRegistry name: %s", record.name.c_str());
        n2r = persis_record_ptr;
    }

    if (record.unversioned_type_id) {
        auto&& res = sd.unversioned_id2reg.emplace(
                record.unversioned_type_id, persis_record_ptr);
        mgb_assert(
                res.second || record.unversioned_type_id ==
                                      dynamic_registry()->unversioned_type_id,
                "duplicated OprRegistry unversioned id: %s",
                std::to_string(record.unversioned_type_id).c_str());
    }
}

const OprRegistry* OprRegistry::find_by_name(const std::string& name) {
    auto&& name2reg = static_data().name2reg;
    auto iter = name2reg.find(name);
    return iter == name2reg.end() ? nullptr : iter->second;
}

const OprRegistry* OprRegistry::find_by_id(size_t id) {
    auto&& id2reg = static_data().id2reg;
    auto iter = id2reg.find(id);
    return iter == id2reg.end() ? nullptr : &iter->second;
}

const OprRegistry* OprRegistry::find_by_type(Typeinfo* type) {
    auto&& type2reg = static_data().type2reg;
    auto iter = type2reg.find(type);
    return iter == type2reg.end() ? nullptr : iter->second;
}

const OprRegistry* OprRegistry::find_by_unversioned_id(size_t unversioned_id) {
    auto&& uid2reg = static_data().unversioned_id2reg;
    auto iter = uid2reg.find(unversioned_id);
    return iter == uid2reg.end() ? nullptr : iter->second;
}

//! find the registry equal to the giving version
const OprRegistryV2* OprRegistryV2::versioned_find_by_id(
        const size_t id, uint8_t version) {
    auto&& id_reg_map = static_data().version_id_reg_map;
    auto iter_version = id_reg_map.find(version);
    if (iter_version != id_reg_map.end()) {
        auto iter = iter_version->second.find(id);
        return iter == iter_version->second.end() ? nullptr : &iter->second;
    }
    return nullptr;
}
//! find the registry equal or below the giving version
const OprRegistryV2* OprRegistryV2::versioned_find_by_typeinfo(
        Typeinfo* type, uint8_t version) {
    const auto& type_reg_map = static_data().version_type_reg_map;
    for (int version_id = version; version_id > 0; version_id--) {
        auto iter_version = type_reg_map.find(version_id);
        if (iter_version != type_reg_map.end()) {
            auto iter = iter_version->second.find(type);
            if (iter == iter_version->second.end()) {
                continue;
            } else {
                return iter->second;
            }
        }
    }
    return nullptr;
}

void OprRegistryV2::versioned_add(
        const OprRegistryV2& record, uint8_t min_version, uint8_t max_version) {
    mgb_assert(max_version >= min_version);

    auto&& sd = static_data();
    auto id = record.type_id;
    uint64_t type_id = id;
    //! record.type->name is nullptr when MGB_VERBOSE_TYPEINFO_NAME==0
#if MGB_VERBOSE_TYPEINFO_NAME
    if (record.type && record.type->name) {
        type_id = MGB_HASH_RUNTIME(std::string(record.type->name));
    }
#endif
    for (uint8_t version = min_version; version <= max_version; version++) {
        auto&& registry_map = sd.version_id_reg_map[version];
        auto versioned_record = record;
        versioned_record.version = version;
        mgb_assert(
                registry_map.find(id) == registry_map.end() ||
                        id == dynamic_registry_v2()->type_id,
                "dduplicated OprRegistryV2 of %s\n", record.name.c_str());
        auto registry_ins = registry_map.emplace(id, versioned_record);
        if (!registry_ins.second) {
            //! the registry is dynamic
            mgb_assert(!record.converter);
            registry_map[id] = versioned_record;
        }
        //! sometimes the register id and the hash typeinfo is not same, just as
        //! dynamic Operator
        if (id != type_id) {
            mgb_assert(
                    registry_map.find(type_id) == registry_map.end(),
                    "duplicated OprRegistryV2 of %s\n", record.name.c_str());
            registry_map.emplace(type_id, versioned_record);
        }
        auto&& registry_type_map = sd.version_type_reg_map[version];
        registry_type_map.emplace(record.type, &registry_map[id]);
    }
}

void OprRegistry::add_using_dynamic_loader(
        Typeinfo* type, const std::string& name, const OprDumper& dumper) {
    // dynamic oprs are implemented by mapping different opr types to the same
    // persist_type_id
    add({type,
         dynamic_registry()->persist_type_id,
         name,
         dumper,
         {},
         {},
         dynamic_registry()->unversioned_type_id});
    mgb_assert(type, "type must be not nullptr");
    OprRegistryV2::versioned_add(
            {type, dynamic_registry_v2()->type_id, type->name, dumper,
             dynamic_registry_v2()->loader, nullptr},
            CURRENT_VERSION, CURRENT_VERSION);
}

#if MGB_ENABLE_DEBUG_UTIL
std::vector<std::vector<std::pair<size_t, std::string>>> OprRegistry::
        dump_registries() {
    auto&& id2reg = static_data().id2reg;
    std::vector<std::vector<std::pair<size_t, std::string>>> result;
    //! version 1 is old register, version 2 is registerV2
    result.resize(CURRENT_VERSION + 1);
    std::vector<std::pair<size_t, std::string>> old_version;
    for (auto iter = id2reg.begin(); iter != id2reg.end(); ++iter) {
        if (iter->second.name.size() == 0)
            old_version.push_back(std::make_pair(iter->first, "<special>"));
        else
            old_version.push_back(std::make_pair(iter->first, iter->second.name));
    }
    result[VERSION_1] = old_version;
    auto&& version_id_reg_map = static_data().version_id_reg_map;
    for (int version_id = CURRENT_VERSION; version_id > 1; version_id--) {
        std::vector<std::pair<size_t, std::string>> version_opr;
        auto&& version_map = version_id_reg_map[version_id];
        for (auto&& it : version_map) {
            if (it.second.name.size() == 0)
                version_opr.push_back(std::make_pair(it.first, "<special>"));
            else
                version_opr.push_back(std::make_pair(it.first, it.second.name));
        }
        result[version_id] = version_opr;
    }
    return result;
}

std::vector<std::vector<std::pair<size_t, std::string>>> OprRegistry::
        recorded_serialized_oprs(bool begin_record, bool end_record) {
    MGB_LOCK_GUARD(static_data().g_record_map_mtx);
    if (begin_record) {
        static_data().recorded = true;
        return {};
    }
    if (end_record) {
        static_data().recorded = false;
        std::vector<std::vector<std::pair<size_t, std::string>>> result;
        result.resize(CURRENT_VERSION + 1);
        auto& recorded = static_data().dumped_opr;
        for (int version_id = CURRENT_VERSION; version_id > 0; version_id--) {
            std::vector<std::pair<size_t, std::string>> version_opr;
            auto&& version_map = recorded[version_id];
            for (auto&& it : version_map) {
                if (it.second.size() == 0)
                    version_opr.push_back(std::make_pair(it.first, "<special>"));
                else
                    version_opr.push_back(std::make_pair(it.first, it.second));
            }
            result[version_id] = version_opr;
        }
        static_data().dumped_opr.clear();
        return result;
    }
    return {};
}

void mgb::serialization::record_opr_dumped(
        const size_t id, std::string name, int version) {
    if (static_data().recorded) {
        MGB_LOCK_GUARD(static_data().g_record_map_mtx);
        auto& opr_dumped = static_data().dumped_opr;
        if (name.size() == 0)
            opr_dumped[version][id] = "<special>";
        else
            opr_dumped[version][id] = name;
    }
}
#else

void mgb::serialization::record_opr_dumped(const size_t, std::string, int) {}
#endif

namespace {
const VarNodeArray& default_accessor(const VarNodeArray& outputs) {
    return outputs;
}
}  // namespace

OprWithOutputAccessor::OprWithOutputAccessor(cg::OperatorNodeBase* opr) : m_opr(opr) {
    m_accessor = &default_accessor;
};
OprWithOutputAccessor::OprWithOutputAccessor(
        cg::OperatorNodeBase* opr, Accessor accessor)
        : OprWithOutputAccessor(opr) {
    if (accessor) {
        m_accessor = accessor;
    }
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
