/**
 * \file src/opr-mm/impl/lock.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/lock.h"

#include <atomic>

using namespace mgb;
using namespace opr;
using namespace intl;

/* ===================== LockBase ===================== */

struct LockBase::LockPool {
    std::mutex mtx;
    struct Entry {
        size_t refcnt = 0;
        std::mutex mtx;
    };
    std::unordered_map<size_t, Entry> id2lock;
};

struct LockBase::LockGroup {
    size_t nr_acquire = 0, nr_release = 0;
    std::atomic_size_t nr_acq_finish{0}, nr_rel_finish{0};
};

class LockBase::LockGroupSet final: public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    public:
        std::mutex mtx;
        std::unordered_map<size_t, LockGroup> id2grp;
};
MGB_TYPEINFO_OBJ_IMPL(LockBase::LockGroupSet);

LockBase::LockPool LockBase::sm_lock_pool;

LockBase::LockBase(const OperatorNodeBaseCtorParam &opr_param,
        VarNode *var, const LockParam &param, Action action):
    Super(opr_param),
    m_param(param), m_action(action)
{
    {
        MGB_LOCK_GUARD(sm_lock_pool.mtx);
        ++ sm_lock_pool.id2lock[param.lock_id].refcnt;
    }
    add_equivalence_component<PODHash<LockParam>>(&m_param);

    add_input({var});
    add_output(None);
}

LockBase::~LockBase() {
    MGB_LOCK_GUARD(sm_lock_pool.mtx);
    if (!-- sm_lock_pool.id2lock.at(m_param.lock_id).refcnt) {
        sm_lock_pool.id2lock.erase(m_param.lock_id);
    }
}

void LockBase::add_input_layout_constraint() {
    auto rst = owner_graph()->current_comp_seq()->user_data().
        get_user_data_or_create<LockGroupSet>();
    {
        MGB_LOCK_GUARD(rst->mtx);
        m_cur_group = &rst->id2grp[m_param.group_id];
    }
    if (m_action == Action::ACQUIRE)
        ++ m_cur_group->nr_acquire;
    else {
        mgb_assert(m_action == Action::RELEASE);
        ++ m_cur_group->nr_release;
    }
}

void LockBase::scn_do_execute_finish(const DeviceTensorND &) {
    std::mutex *lock;
    {
        MGB_LOCK_GUARD(sm_lock_pool.mtx);
        lock = &sm_lock_pool.id2lock[m_param.lock_id].mtx;
    }
    auto grp = m_cur_group;

    mgb_throw_if(!grp->nr_acquire || !grp->nr_release, GraphError,
            "lock acquire/release mismatch");

    if (m_action == Action::ACQUIRE) {
        size_t nr = ++ grp->nr_acq_finish;
        mgb_assert(nr <= grp->nr_acquire);
        if (nr == grp->nr_acquire) {
            lock->lock();
            grp->nr_acq_finish.store(0);
        }
    } else {
        size_t nr = ++ grp->nr_rel_finish;
        mgb_assert(nr <= grp->nr_release);
        if (nr == grp->nr_release) {
            lock->unlock();
            grp->nr_rel_finish.store(0);
        }
    }
}

/* ===================== LockMaker ===================== */

template<typename Opr>
SymbolVar LockMaker<Opr>::make(SymbolVar var, const LockParam &param,
        const OperatorNodeConfig &config) {
    return var.insert_single_output_opr<Opr>(var.node(), param, config);
}

/* ===================== LockImpl ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LockAcquire);
LockAcquire::LockAcquire(VarNode *var, const LockParam &param,
        const OperatorNodeConfig &config):
    Super({var->owner_graph(), config, "lock_acquire", {var}},
            var, param, Action::ACQUIRE)
{
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LockRelease);
LockRelease::LockRelease(VarNode *var, const LockParam &param,
        const OperatorNodeConfig &config):
    Super({var->owner_graph(), config, "lock_release", {var}},
            var, param, Action::RELEASE)
{
}


/* ===================== explicit instantialization ===================== */
namespace mgb {
namespace opr {
namespace intl {
    template class LockMaker<LockAcquire>;
    template class LockMaker<LockRelease>;
}
}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

