/**
 * \file src/serialization/include/megbrain/serialization/opr_registry.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace serialization {
    // forward declaration
    class OprDumpContext;
    class OprLoadContext;
    class OprShallowCopyContext;

    //! dump opr internal params to OprDumpContext
    using OprDumper = thin_function<void(
            OprDumpContext &ctx, const cg::OperatorNodeBase &opr)>;

    //! load and restore operator from OprLoadContext
    using OprLoader = thin_function<cg::OperatorNodeBase*(
            OprLoadContext &ctx, const cg::VarNodeArray &inputs,
            const OperatorNodeConfig &config)>;

    //! shallow copy function for a single operator
    using OprShallowCopy = thin_function<cg::OperatorNodeBase*(
            const OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr, const VarNodeArray &inputs,
            const OperatorNodeConfig &config)>;

    //! record of a single operator
    struct OprRegistry {
        Typeinfo *type;
        uint64_t persist_type_id;
        std::string name;
        OprDumper dumper;
        OprLoader loader;
        OprShallowCopy shallow_copy; //!< set to empty to use default impl
        uint64_t unversioned_type_id;

        static void add(const OprRegistry &record);

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
        static void add_using_dynamic_loader(
                Typeinfo *type, const std::string &name,
                const OprDumper &dumper);

        //! find registry by opr type name; return nullptr if not found
        static const OprRegistry* find_by_name(const std::string &name);

        //! find registry by persist_type_id; return nullptr if not found
        static const OprRegistry* find_by_id(size_t id);

        //! find registry by type; return nullptr if not found
        static const OprRegistry* find_by_type(Typeinfo *type);

        // TODO: This is hack. Refactor this out.
        //! Find registry by unversioned id; return nullptr if not found
        static const OprRegistry* find_by_unversioned_id(size_t unversioned_id);

#if MGB_ENABLE_DEBUG_UTIL
        //! dump registered oprs
        static std::vector<std::pair<uint64_t, std::string>> dump_registries();
#endif
    };

} // namespace serialization
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
