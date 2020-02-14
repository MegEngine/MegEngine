/**
 * \file src/plugin/include/megbrain/plugin/profiler.h
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
#include "megbrain/plugin/base.h"
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/utils/small_vector.h"
#include "megbrain/utils/timer.h"

#if MGB_ENABLE_JSON

#include <map>
#include <memory>
#include <thread>
#include <unordered_map>

namespace opr_profile {
class OprProfileHolder final : public mgb::UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

public:
    mgb::ThinHashMap<mgb::cg::OperatorNodeBase*,
                     std::shared_ptr<mgb::json::Value>>
            id2object_map;
};
}  // namespace opr_profile

namespace mgb {
/*!
 * \brief graph profiler for operators
 */
class GraphProfiler final : public PluginBase {
    //! time of host event relative to some specific starting point
    using CompNodeEventPtr = std::unique_ptr<CompNode::Event>;
    struct OprHostTime {
        double start = -1,  //!< start of opr on each dispatch thread
                kern = -1,  //!< first start of kern on each dispatch thread
                end = -1;   //!< opr end time on each dispatch thread
    };
    struct OprKernEvent {
        CompNodeEventPtr start,  //!< opr starts, recorded for all comp nodes
                kern,            //!< start for kernels on a comp node
                end;             //!< end of kernels on a comp node
    };

    //! comp nodes used in current compiled function
    const CompNode::UnorderedSet* m_used_comp_node = nullptr;

    //! (opr, dispatch thread) => host time
    std::unordered_map<std::pair<cg::OperatorNodeBase*, std::thread::id>,
                       OprHostTime, pairhash>
            m_host_time;

    //! (opr, comp node) => kern event
    std::unordered_map<std::pair<cg::OperatorNodeBase*, CompNode>, OprKernEvent,
                       pairhash>
            m_kern_event;

    //! (opr) => computation and memory usage
    using OprFootprintRst = OprFootprint::Result;
    std::unordered_map<cg::OperatorNodeBase*, OprFootprintRst> m_opr_fp_rst;

    std::unique_ptr<OprFootprint> m_opr_footprint_ptr{
            std::make_unique<OprFootprint>()};

    //! first event on each comp node
    Maybe<CompNode::UnorderedMap<CompNodeEventPtr>> m_start_of_time;
    std::mutex m_mtx;
    RealTimer m_timer;

    //! return whether given opr should be profiled
    bool opr_filter(cg::OperatorNodeBase* opr);

    void ensure_start_time();
    void record_event(CompNodeEventPtr& dest, CompNode comp_node);

public:
    GraphProfiler(cg::ComputingGraph* graph);
    ~GraphProfiler() noexcept;

    /*!
     * \brief convert only profiling result to json
     */
    std::shared_ptr<json::Object> to_json() const;

    /*!
     * \brief dump to visualizer format
     */
    std::shared_ptr<json::Object> to_json_full(
            cg::AsyncExecutable* graph_exec) const {
        return json::Object::make({{"graph_exec", graph_exec->to_json()},
                                   {"profiler", to_json()}});
    }
};

}  // namespace mgb

#endif  // MGB_ENABLE_JSON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
