#pragma once

#include <list>
#include <map>
#include <memory>
#include <vector>

#include "megbrain/common.h"
#include "megbrain/imperative/subgraph.h"
#include "megbrain/imperative/utils/allocator.h"
#include "megbrain/imperative/utils/local_ptr.h"
#include "megbrain/imperative/utils/span.h"

namespace mgb {
namespace imperative {

class ValueRef;
class ValueRefList;
class Operator;
class Transformation;

/**
 * \brief args of dispatch action
 *
 */
struct TransformationFrame {
    const Operator& op;
    const Span<ValueRef>& inputs;
};

struct TransformationContext {
    std::vector<std::shared_ptr<Transformation>> transformations;
    std::vector<std::string> scopes;
    // TODO: deprecate TransformationGuard, let next_transformation == frames.size()
    size_t next_transformation = 0;
    std::vector<TransformationFrame> frames;
    ForwardAllocator<ValueRef> allocator;
};

/**
 * \brief Transformation handles operation requests.
 *
 * There is an transformation stack in each context. When user send an operation
 * request, it is firstly passed to the top transformation. When a transformation in the
 * stack receiving a request, it should handle it and give a response. Transformations
 * are allowed to send requests when handling other requests, those requests would be
 * sent to downstairs. A transformation can only be added to one stack.
 */
class Transformation : public std::enable_shared_from_this<Transformation> {
public:
    using pos_t =
            decltype(std::declval<TransformationContext>().transformations)::iterator;

    class TransformationGuard {
    private:
        size_t m_priority;

    public:
        TransformationGuard(size_t priority) : m_priority{priority} {
            auto& context = get_context();
            std::swap(m_priority, context.next_transformation);
            mgb_assert(
                    context.next_transformation <= context.transformations.size(),
                    "invalid priority: %zu vs %zu", context.next_transformation,
                    context.transformations.size());
        }
        ~TransformationGuard() {
            std::swap(m_priority, get_context().next_transformation);
        }
    };

private:
    size_t m_priority = std::numeric_limits<size_t>::max();

public:
    /**
     * \brief handle a dispatch request
     *
     * \param op
     * \param inputs
     * \return ValueRefList
     */
    virtual ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) = 0;

    virtual ValueRef unwrap(ValueRef value) = 0;

    virtual std::string name() const = 0;

    /**
     * \brief called when added to a stack.
     */
    virtual void on_register(){};

    /**
     * \brief called when remove from a stack.
     *
     * Some transformations, like GradTransformation and TraceTransformation, produce
     * special values when handling requests. Thus they should recover these values on
     * unregistering because other transformations cann't recognize them.
     */
    virtual void on_unregister() noexcept {};

public:
    static auto top() { return get_context().transformations.begin(); }
    static auto bottom() { return get_context().transformations.end(); }
    static void push_scope(std::string scope) { get_context().scopes.push_back(scope); }
    static void pop_scope(std::string scope) {
        auto& context = get_context();
        auto top = context.scopes.back();
        context.scopes.pop_back();
        mgb_assert(top == scope);
    }
    static std::vector<std::string> scopes() { return get_context().scopes; }

    /**
     * \brief position at transformation stack
     *
     * \return auto position
     */
    auto pos() const {
        mgb_assert(
                m_priority != std::numeric_limits<size_t>::max(), "not yet registered");
        return top() + m_priority;
    }

    /**
     * \brief register this at given position
     *
     * \param pos position
     */
    void register_at(pos_t pos) {
        auto& context = get_context();
        mgb_assert(
                m_priority == std::numeric_limits<size_t>::max(), "already registered");
        size_t priority = pos - context.transformations.begin();
        for (auto iter = pos; iter != context.transformations.end(); ++iter) {
            iter->get()->m_priority++;
        }
        m_priority = priority;
        context.transformations.insert(pos, shared_from_this());
        {
            TransformationGuard _{m_priority + 1};
            on_register();
        }
        // assert priority
    }

    /**
     * \brief unregister this from transformation stack
     */
    void unregister() noexcept {
        auto& context = get_context();
        mgb_assert(
                m_priority != std::numeric_limits<size_t>::max(), "not yet registered");
        {
            TransformationGuard _{m_priority + 1};
            on_unregister();
        }
        size_t priority = m_priority;
        auto pos = top() + priority;
        for (auto iter = pos; iter != context.transformations.end(); ++iter) {
            iter->get()->m_priority--;
        }
        m_priority = std::numeric_limits<size_t>::max();
        context.transformations.erase(pos);
        // TODO: assert priority
    }
    // FIXME: deprecated
    [[nodiscard]] TransformationGuard current_level_guard() { return m_priority; }

    /**
     * \brief swap current context with target
     *
     * \param context target context
     */
    static void swap_context(TransformationContext& context) {
        auto& current_context = get_context();
        std::swap(context.transformations, current_context.transformations);
        std::swap(context.scopes, current_context.scopes);
        std::swap(context.next_transformation, current_context.next_transformation);
        std::swap(context.allocator, current_context.allocator);
    }

    static TransformationContext& get_context();

    friend ValueRefList apply(const Operator& op, Span<ValueRef> inputs);
    friend class ValueRef;
};

}  // namespace imperative
}  // namespace mgb
