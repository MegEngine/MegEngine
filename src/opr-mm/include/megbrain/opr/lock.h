/**
 * \file src/opr-mm/include/megbrain/opr/lock.h
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
#include "megbrain/opr/internal/identical_fwd.h"

namespace mgb {
namespace opr {

namespace intl {
    MGB_DEFINE_CLS_WITH_SUPER(LockBase, ForwardInputToOutput) // {
        public:
            enum Action {
                ACQUIRE, RELEASE
            };

            /*!
             * lock works on a group of vars; when all vars in a group are
             * produced, a global lock indexed by lock_id would be acquired
             *
             * lock_id is global, but group is limited to a comp graph
             */
            struct LockParam {
                size_t lock_id, group_id;
            };

            LockBase(const OperatorNodeBaseCtorParam &opr_param,
                    VarNode *var, const LockParam &param, Action action);
            ~LockBase();

        private:
            struct LockPool;
            struct LockGroup;
            class LockGroupSet;
            static LockPool sm_lock_pool;
            const LockParam m_param;
            Action m_action;
            LockGroup *m_cur_group = nullptr;

            void add_input_layout_constraint() override;
            void scn_do_execute_finish(const DeviceTensorND &val) override;
    };

    template<typename Opr>
    MGB_DEFINE_CLS_WITH_SUPER(LockMaker, LockBase) // {
        protected:
            using Super::Super;
        public:
            static SymbolVar make(
                    SymbolVar var,
                    const LockParam &param,
                    const OperatorNodeConfig &config = {});
    };
}

/*!
 * \brief acquire a global lock when all vars in the group are ready
 */
MGB_DEFINE_OPR_CLASS(LockAcquire, intl::LockMaker<LockAcquire>) // {
    public:
        LockAcquire(VarNode *var, const LockParam &param,
                const OperatorNodeConfig &config);
};

/*!
 * \brief release the global lock when all vars in the group are ready
 */
MGB_DEFINE_OPR_CLASS(LockRelease, intl::LockMaker<LockRelease>) // {
    public:
        LockRelease(VarNode *var, const LockParam &param,
                const OperatorNodeConfig &config);
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}


