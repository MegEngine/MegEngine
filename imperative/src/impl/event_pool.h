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
