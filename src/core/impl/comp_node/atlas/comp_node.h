/**
 * \file src/core/impl/comp_node/atlas/comp_node.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include <mutex>
#include "../impl_helper.h"

namespace mgb {
class AtlasCompNode final : public CompNodeImplHelper {
public:
    static constexpr Flag sm_flag = Flag::QUEUE_LIMITED |
                                    Flag::HAS_COPY_STREAM |
                                    Flag::SUPPORT_UNIFIED_ADDRESS;

    class CompNodeImpl;
    class EventImpl;

    //! whether cuda comp node is available
    static bool available();

    static void foreach (thin_function<void(CompNode)> callback);
    static void finalize();
    static size_t get_device_count();
    static Impl* load_atlas(const Locator& locator,
                            const Locator& locator_logical);
    static void sync_all();

};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
