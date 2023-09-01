#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace serialization {
// forward declaration
class OprDumpContext;
class OprLoadContext;
class OprShallowCopyContext;
class OprWithOutputAccessor {
    cg::OperatorNodeBase* m_opr;
    using Accessor = thin_function<const VarNodeArray(const VarNodeArray&)>;
    Accessor m_accessor;

public:
    OprWithOutputAccessor(cg::OperatorNodeBase* opr);
    OprWithOutputAccessor(cg::OperatorNodeBase* opr, Accessor accessor);
    VarNode* output(size_t idx) const { return output().at(idx); }
    VarNodeArray output() const { return m_accessor(m_opr->output()); }
    VarNodeArray usable_output() const { return m_accessor(m_opr->usable_output()); }
    cg::OperatorNodeBase* opr() { return m_opr; }
};

void record_opr_dumped(const size_t id, std::string name, int version);

//! dump opr internal params to OprDumpContext
using OprDumper =
        thin_function<void(OprDumpContext& ctx, const cg::OperatorNodeBase& opr)>;

//! load and restore operator from OprLoadContext
//! is also used by GraphLoadConfig.
using OprLoader = thin_function<cg::OperatorNodeBase*(
        OprLoadContext& ctx, const cg::VarNodeArray& inputs,
        const OperatorNodeConfig& config)>;

//! loader that can change opr output map for compatibility
using OprLoaderWrapper = thin_function<OprWithOutputAccessor(
        OprLoadContext& ctx, const cg::VarNodeArray& inputs,
        const OperatorNodeConfig& config)>;

//! shallow copy function for a single operator
using OprShallowCopy = thin_function<cg::OperatorNodeBase*(
        const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
        const VarNodeArray& inputs, const OperatorNodeConfig& config)>;

//! Convert some modified Opr to compatible Opr
using OprConvertToCompatible = thin_function<cg::OperatorNodeBase*(
        cg::OperatorNodeBase*, const VarNodeArray&)>;

//! record of a single operator
struct OprRegistry {
    Typeinfo* type;
    uint64_t persist_type_id;
    std::string name;
    OprDumper dumper;
    OprLoaderWrapper loader;
    OprShallowCopy shallow_copy;  //!< set to empty to use default impl
    uint64_t unversioned_type_id;
    OprConvertToCompatible converter = nullptr;

    MGE_WIN_DECLSPEC_FUC static void add(const OprRegistry& record);
    /*!
     * \brief register an operator to use dynamic loader
     *
     * The dumper should write a string using
     * OprDumpContext::dump_buf_with_len(); the string would be used as
     * operator id during loading, and actual loader is obtained by
     * OprLoadContext::make_opr_loader().
     *
     * See TestSerializer.DynamicLoader for an example
     */
    MGE_WIN_DECLSPEC_FUC static void add_using_dynamic_loader(
            Typeinfo* type, const std::string& name, const OprDumper& dumper);

    //! find registry by opr type name; return nullptr if not found
    MGE_WIN_DECLSPEC_FUC static const OprRegistry* find_by_name(
            const std::string& name);

    //! find registry by persist_type_id; return nullptr if not found
    MGE_WIN_DECLSPEC_FUC static const OprRegistry* find_by_id(size_t id);

    //! find registry by type; return nullptr if not found
    MGE_WIN_DECLSPEC_FUC static const OprRegistry* find_by_type(Typeinfo* type);

    // TODO: This is hack. Refactor this out.
    //! Find registry by unversioned id; return nullptr if not found
    MGE_WIN_DECLSPEC_FUC static const OprRegistry* find_by_unversioned_id(
            size_t unversioned_id);

#if MGB_ENABLE_DEBUG_UTIL
    //! dump registered oprs
    MGE_WIN_DECLSPEC_FUC static std::vector<std::vector<std::pair<size_t, std::string>>>
    dump_registries();
    //! record all dumped/loaded oprs (hash_id --> type)
    MGE_WIN_DECLSPEC_FUC static std::vector<std::vector<std::pair<size_t, std::string>>>
    recorded_serialized_oprs(bool begin_record, bool end_record);
#endif
};

//! record of a single operator
struct OprRegistryV2 {
    Typeinfo* type;
    uint64_t type_id;
    std::string name;
    OprDumper dumper;
    OprLoaderWrapper loader;
    OprConvertToCompatible converter;
    uint8_t version = 2;

    MGE_WIN_DECLSPEC_FUC uint8_t get_version() const { return version; }

    //! register opr load/dump to version2regmap
    MGE_WIN_DECLSPEC_FUC static void versioned_add(
            const OprRegistryV2& record, uint8_t min_version, uint8_t max_version,
            bool dynamic = false);

    MGE_WIN_DECLSPEC_FUC static const OprRegistryV2* versioned_find_by_id(
            const size_t id, uint8_t version);

    MGE_WIN_DECLSPEC_FUC static const OprRegistryV2* versioned_find_by_typeinfo(
            Typeinfo* type, uint8_t version);
};

}  // namespace serialization
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
