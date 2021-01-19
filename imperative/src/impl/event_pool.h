/**
 * \file imperative/src/impl/event_pool.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"

namespace mgb {
namespace imperative {

class EventPool : CompNodeDepedentObject {
    CompNode::UnorderedMap<CompNode::EventPool> m_cn2pool;
    Spinlock m_lock;
    size_t m_flags;

    EventPool(size_t flags = 0);

public:
    static EventPool& with_timer();
    static EventPool& without_timer();
    CompNode::Event* alloc(CompNode cn);
    std::shared_ptr<CompNode::Event> alloc_shared(CompNode cn);
    void free(CompNode::Event* event);
    std::shared_ptr<void> on_comp_node_finalize();
    ~EventPool();
};
}  // namespace imperative
}  // namespace mgb
