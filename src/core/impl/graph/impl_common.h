/**
 * \file src/core/impl/graph/impl_common.h
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
#include "./static_infer_impl.h"

#include <unordered_set>

namespace mgb {
namespace cg {

    class ComputingGraphImpl;

    /*!
     * \brief extra info for comp seq
     *
     * This is stored in the ComputingSequence object associated with a graph.
     */
    struct CompSeqExtraInfo {
        ThinHashMap<const VarNode *, ComputingGraph::VarReceiverInfo>
            var2recvinfo;

        //! target tags for shape/value infer; setup by topo sorter and used by
        //! CompSeqManager::reset_dest
        ThinHashSet<static_infer::StaticInferManagerImpl::TagHandler*>
            infer_dest;


        //! missing inputs, initialized by CompSeqManager::reset_dest()
        VarNodeSet missing_for_shape, missing_for_value;

        //! source nodes needed for static infer; may contain nodes not in
        //! computing sequence; initialized by CompSeqManager::reset_dest()
        static_infer::DepVal rt_static_infer_src;
    };

} // namespace cg
} // namespace mgb


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

