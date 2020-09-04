#include "./event_pool.h"

namespace mgb {
namespace imperative {

EventPool::EventPool(size_t flags) : m_flags{flags} {}

EventPool& EventPool::with_timer() {
    static Spinlock lock;
    static std::unique_ptr<EventPool> ptr;
    MGB_LOCK_GUARD(lock);
    if (!ptr || ptr->is_finalized()) {
        ptr.reset(new EventPool(CompNode::Event::NEED_TIMER));
    }
    return *ptr;
}
EventPool& EventPool::without_timer() {
    static Spinlock lock;
    static std::unique_ptr<EventPool> ptr;
    MGB_LOCK_GUARD(lock);
    if (!ptr || ptr->is_finalized()) {
        ptr.reset(new EventPool());
    }
    return *ptr;
}
CompNode::Event* EventPool::alloc(CompNode cn) {
    CompNode::EventPool* pool;
    {
        MGB_LOCK_GUARD(m_lock);
        auto iter = m_cn2pool.find(cn);
        if (iter == m_cn2pool.end()) {
            iter = m_cn2pool
                           .emplace(std::piecewise_construct,
                                    std::forward_as_tuple(cn),
                                    std::forward_as_tuple(cn, m_flags))
                           .first;
        }
        pool = &iter->second;
    }
    return pool->alloc();
}
std::shared_ptr<CompNode::Event> EventPool::alloc_shared(CompNode cn) {
    auto* raw_event = alloc(cn);
    return {raw_event, [this](CompNode::Event* event){ this->free(event); }};
}
void EventPool::free(CompNode::Event* event) {
    CompNode::EventPool* pool;
    {
        MGB_LOCK_GUARD(m_lock);
        pool = &m_cn2pool.at(event->comp_node());
    }
    pool->free(event);
}
std::shared_ptr<void> EventPool::on_comp_node_finalize() {
    MGB_LOCK_GUARD(m_lock);
    for (auto&& i : m_cn2pool) {
        i.second.assert_all_freed();
    }
    return {};
}
EventPool::~EventPool() {
    for (auto&& i : m_cn2pool) {
        i.second.assert_all_freed();
    }
}

}  // namespace imperative
}  // namespace mgb
