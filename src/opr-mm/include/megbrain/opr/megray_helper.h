/**
 * \file src/opr-mm/include/megbrain/opr/megray_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/metahelper.h"
#include "megbrain/opr/group_manager.h"
#include "megray.h"

namespace mgb {
namespace opr {

/*!
 * gather MegRay unique ids and build communicator, use hash for deduplication
 */
class MegRayCommunicatorBuilder final : public mgb::UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    private:
        std::unordered_map<uint64_t, std::shared_ptr<MegRay::Communicator>> m_megray_comms;

    public:
        std::shared_ptr<MegRay::Communicator> get_megray_comm(
                uint64_t hash, std::string key, uint32_t size, uint32_t rank,
                MegRay::Backend backend,
                std::shared_ptr<mgb::opr::GroupClient> group_client);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
