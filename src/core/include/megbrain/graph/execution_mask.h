/**
 * \file src/core/include/megbrain/graph/execution_mask.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/operator_node.h"

#if MGB_ENABLE_COND_EXEC

namespace mgb {

namespace cg {

/*!
 * \brief a mask object to be associated with an operator to indicate whether it
 *      should be executed
 */
class ExecutionMask final : public std::enable_shared_from_this<ExecutionMask>,
                            NonCopyableObj {
    MGB_TYPEINFO_OBJ_DECL;

    class RefHolder;

    static std::atomic_size_t sm_tot_id, sm_alive_inst;

    bool m_enabled = false;
    const size_t m_id;
    VarNode* const m_owner;
    ExecutionMask* m_parent = nullptr;
    size_t m_level = 0;  //!< nested level
    SmallVector<ExecutionMask*> m_nested;

public:
    //! the owner can be null (in the case of user-created ExecutionMask)
    explicit ExecutionMask(VarNode* owner);

    ~ExecutionMask();

    //! whether there is any instance of ExecutionMask; useful for fast skipping
    //! of code that processes conditional execution
    static bool have_alive_instance() {
        return sm_alive_inst.load(std::memory_order_relaxed);
    }

    VarNode* owner() const { return m_owner; }

    /*!
     * \brief register this ExecutionMask to an operator
     *
     * This ExecutionMask must be managed by a std::shared_ptr. A reference
     * would be kept in the computing graph. Only one mask can be registered to
     * an operator.
     */
    void register_to_opr(OperatorNodeBase* opr);

    bool enabled() const { return m_enabled; }

    /*!
     * \brief set enable flag
     *
     * Note: if flag is false and there are nested ExecutionMask objects, they
     * would all be disabled.
     */
    void enable(bool flag);

    //! add a nested ExecutionMask
    void add_nested(ExecutionMask* nested);

    //! the mask who contains this one as nested
    ExecutionMask* parent() const { return m_parent; }

    //! get a non-zero global ID for this mask (it can be used for printing)
    size_t id() const { return m_id; }

    //! get the ExecutionMask associated with an opr, or nullptr
    static ExecutionMask* get_from_opr(const OperatorNodeBase* opr) {
        return opr->node_prop().attribute().accessory.exec_mask;
    }

    /*!
     * \brief get the one with lowest level (i.e. as nested in another) of a and
     *      b
     *
     * Nullptr would be returned if they are not directly nested (i.e. one is
     *  not the ancestor of another)
     *
     * The params can be null, which represent the root level.
     */
    static ExecutionMask* find_direct_lowest(ExecutionMask* a,
                                             ExecutionMask* b);
};

}  // namespace cg
}  // namespace mgb

#endif  // MGB_ENABLE_COND_EXEC

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
