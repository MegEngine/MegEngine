/**
 * \file src/core/include/megbrain/graph/bases.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/json.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/exception.h"
#include "megbrain/comp_node.h"

#include <string>

#ifndef MGB_ENABLE_SUBLINEAR
#define MGB_ENABLE_SUBLINEAR ((!MGB_BUILD_SLIM_SERVING) && (!!MGB_HAVE_THREAD))
#endif  //  MGB_ENABLE_SUBLINEAR

#ifndef MGB_ENABLE_MEMORY_SWAP
#define MGB_ENABLE_MEMORY_SWAP \
    ((!MGB_BUILD_SLIM_SERVING) && (!!MGB_HAVE_THREAD) && (MGB_CUDA))
#endif

#ifndef MGB_ENABLE_PARTIAL_EXECUTION
#define MGB_ENABLE_PARTIAL_EXECUTION (!MGB_BUILD_SLIM_SERVING)
#endif  //  MGB_ENABLE_PARTIAL_EXECUTION

#ifndef MGB_ENABLE_COND_EXEC
#define MGB_ENABLE_COND_EXEC !MGB_BUILD_SLIM_SERVING
#endif
#if MGB_ENABLE_COND_EXEC
#define MGB_IF_COND_EXEC(x...) x
#else
#define MGB_IF_COND_EXEC(x...)
#endif

namespace mgb {

//! computing graph
namespace cg {

namespace static_infer {
    struct DepElement;
};

class GraphError: public MegBrainError {
    public:
        using MegBrainError::MegBrainError;
};

class OperatorNodeBase;
class ComputingGraph;

/*!
 * \brief Base class for a node in the graph.
 *
 * Each node must have a name for debugging and graph dump, and each node is
 * uniquely identified by its memory address. Every node in a computing graph
 * has its unique numerical ID.
 */
class GraphNodeBase: public json::Serializable, public NonCopyableObj {
    ComputingGraph* const m_owner_graph;
    size_t m_id;

    protected:
        ~GraphNodeBase() = default;

    public:
        GraphNodeBase(ComputingGraph *owner_graph);

        ComputingGraph* owner_graph() const {
            return m_owner_graph;
        }

        //! get node ID as string
        std::string id_str() const {
            return std::to_string(m_id);
        }

        //! get node ID as number
        size_t id() const {
            return m_id;
        }
};

/*!
 * \brief an object that executes asynchronously
 */
class AsyncExecutable : public json::Serializable,
                        public CompNodeDepedentObject {
    UserDataContainer m_user_data;

    public:
        virtual ~AsyncExecutable() noexcept;

        virtual AsyncExecutable& execute() = 0;

        /*!
         * \brief wait for current task to finish
         */
        virtual AsyncExecutable& wait() = 0;

        /*!
         * \brief previous execution time in seconds
         */
        virtual double get_prev_exec_time() const = 0;

        /*!
         * \brief iterate over operator sequence
         * \param cb callback function, return false to stop iterating
         */
        virtual AsyncExecutable& iter_opr_seq(
                thin_function<bool(OperatorNodeBase*)> cb) = 0;

        /*!
         * \brief get RT_STATIC deps needed for static infer in this func
         */
        virtual const SmallVector<static_infer::DepElement>&
            get_rt_static_source_deps() = 0;

        /*!
         * \brief number of calls to execute()
         */
        virtual size_t get_run_id() const = 0;

        /*!
         * \brief update static memory allocation plan and allocation size
         *
         * Note: as a side effect, static shape inference would be executed and
         * var shapes are updated.
         *
         * \return static allocation size for each comp node
         */
        virtual const CompNode::UnorderedMap<size_t>&
        update_static_alloc_plan_and_get_size() = 0;

        /*!
         * \brief clear device memory; memory would be allocated in the next run
         */
        virtual void clear_device_memory() = 0;

        //! get the graph that owns this executable; nullptr if no owner graph
        virtual ComputingGraph* owner_graph() const = 0;

        //! user data associated with a compiled executable
        UserDataContainer& user_data() {
            return m_user_data;
        }
};


} // namespace cg
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

