/**
 * \file src/core/include/megbrain/graph/cg.h
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
#include "megbrain/graph/symbol_var.h"
#include "megbrain/graph/static_infer.h"
#include "megbrain/graph/seq_comp_node_opt.h"
#include "megbrain/utils/event.h"
#include "megbrain/system.h"

#if MGB_ENABLE_JSON
#include "megbrain/utils/json.h"
#endif

namespace mgb {
namespace cg {

/*!
 * \brief allocation strategy for device storage in computing graphs
 *
 * Note: all the \p graph params would be NULL for requests originating from
 * ComputingGraph::prealloc_static_storage. Otherwise they are not NULL.
 *
 * This base class already provides an implementation using memory management on
 * the comp node. Sub-classes can override only the methods of interest.
 */
class DeviceMemoryAllocator {
public:
    //! a version sentinel value that should never be returned by
    //! static_alloc_version()
    static constexpr size_t VERSION_INVALID = ~static_cast<size_t>(0);

    virtual ~DeviceMemoryAllocator() = default;

    /*!
     * \brief implement the allocation strategy for static graph-wise storage
     * \param[in] graph the computing graph that requests the memory
     * \param[out] dest output tensor storage; its comp node has been
     *      initialized to target comp node
     */
    virtual void alloc_static(ComputingGraph* graph, DeviceTensorStorage& dest,
                              size_t size);

    /*!
     * \brief implement the allocation strategy for dynamic storage of a
     *      variable
     * \param[in] var the variable that needs memory
     *
     * Note: if allocation fails, MemAllocError should be raised so
     * VarDevMemDefragmenter can catch the error and do defragmentation.
     */
    virtual void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest,
                               size_t size);

    /*!
     * \brief Ensure a contiguous storage for memory defragmenter
     *
     * When doing memory-defragmentation, it is useful to ensure that following
     * allocation requests can be placed in a contiguous storage. This function
     * would be called before calling alloc_dynamic() on the individual vars.
     */
    virtual void defrag_prealloc_contig(ComputingGraph* graph,
                                        CompNode comp_node, size_t size);

    /*!
     * \brief version of static allocation strategy
     *
     * If version changes before graph exec, static memory would be reallocated.
     * This function would be only called once in each graph execution.
     */
    virtual size_t static_alloc_version(ComputingGraph* graph) const;
};

/*!
 * \brief Computing graph.
 *
 * A computing graph manages operators and variables. It can be compiled to
 * create an AsyncExecutable that computs given variables.
 */
class ComputingGraph : public std::enable_shared_from_this<ComputingGraph>,
                       public CompNodeDepedentObject {
    public:
        ComputingGraph();
        virtual ~ComputingGraph() = default;

        /*!
         * \brief graph ID
         *
         * Each graph would be assigned a unique increasing ID; useful for
         * debugging
         */
        size_t id() const {
            return m_id;
        }

        static std::shared_ptr<ComputingGraph> make();

        //! assert that refcnt for ptr is one and destories the ptr
        static void assert_destroy(std::shared_ptr<ComputingGraph>& ptr);

        /*!
         * \brief callback to be invoked when some output is ready
         *
         * note that the output may be deallocated after the call returns if no
         * further node depends on the output
         */
        using Callback = thin_function<void(DeviceTensorND&)>;

        //! specify the callback of one output var
        using OutputSpecItem = std::pair<SymbolVar, Callback>;

        /*!
         * specified what ouptputs are required in compile(); the callback could
         * be empty, to ensure that the var is computed
         */
        using OutputSpec = std::vector<OutputSpecItem>;

        /*!
         * \brief information on how a var is needed by others
         */
        struct VarReceiverInfo;

        /*!
         * \brief generate an executable object that when executed, would call
         *      the callbacks on the output values
         *
         * Also note that only the most recent compiled function could be used,
         * since oprs may have internal state
         */
        virtual std::unique_ptr<AsyncExecutable> compile(
                const OutputSpec &out_spec) = 0;

        /*!
         * \brief compile multiple graph parts for partial execution
         *
         * The parts in \p out_specs correspond to the execution steps of this
         * graph. The returned AsyncExecutable objects should be called in the
         * same order of parts given here.
         *
         * The created AsyncExecutable objects would belong to newly generated
         * graphs (not this graph). So functions compiled by compile() and
         * compile_multi_part() can co-exist. All the new graphs would share
         * device memory with this graph.
         */
        virtual SmallVector<std::unique_ptr<AsyncExecutable>>
        compile_multi_part(const SmallVector<OutputSpec>& out_specs) = 0;

        /*!
         * \brief insert a new operator node; its input must exist in current
         *      graph
         * \return the node in the graph (maybe another node due to
         *      deduplication)
         */
        virtual OperatorNodeBase* insert_opr(
                std::unique_ptr<OperatorNodeBase> opr) = 0;

        /*!
         * \brief get current computing sequence
         */
        virtual AsyncExecutable* current_comp_seq() = 0;

        /*!
         * \brief get information on how a variable is needed in current comp
         *      seq
         */
        virtual const VarReceiverInfo& var_receiver_in_current_comp_seq(
                const VarNode *var) const = 0;

        virtual std::string get_mem_allocation_info() const = 0;

        /*!
         * \brief find var node by its ID
         *
         * Note: this searches recursively in subgraphs, and its complexity is
         * linear with respect to number of vars (there is no indexing on var
         * node ID)
         *
         * \return VarNode pointer if it is found, or nullptr if no var is
         *      found to have equal ID
         */
        virtual VarNode* find_var_by_id(size_t id) const = 0;

        /*!
         * \brief get underlying event connector
         */
        SyncEventConnecter& event() {
            return m_event;
        }

        const SyncEventConnecter& event() const {
            return m_event;
        }

        struct Options {
            //! attribute for a specific operator
            struct OprAttribute {
#if MGB_ENABLE_SUBLINEAR
                /*!
                 * if any opr is in this set, then the split of blocks can only
                 * happen on those oprs.
                 */
                ThinHashSet<OperatorNodeBase*>
                    sublinear_memory_endpoint;

                bool get_sublinear_memory_endpoint(OperatorNodeBase *opr) const
                { return sublinear_memory_endpoint.count(opr); }
#endif
            } opr_attribute;

            //! sequence compile optimization options
            struct SeqOpt {
                //! whether to enable memory forwarding to optimize mem plans
                bool enable_mem_plan_opt = true;

                //! whether to enable static memory reuse (i.e. using optimized
                //! static memory allocation algorithm)
                bool enable_mem_reuse_alloc = true;

                //! whether to enable comp node optimization (e.g. using copy
                //! stream for I/O operators)
                bool enable_seq_comp_node_opt = true;
            } seq_opt;

            //! graph optimization options
            struct GraphOpt {
                //! whether to enable JIT; JIT would also be enabled at O3
                //! this value indicates JIT level: 1 for basic elemwise opr; 2
                //! for including reduce oprs
                uint8_t jit = 0;
                //! whether to enable fine-grained TensorRT opr replace
                bool tensorrt = false;
                //! whether to enable fast-run profiled winograd opr replace
                bool winograd_transform = false;
                //! whether to enable nchw4->chwn4 opr replace
                bool enable_chwn4 = false;
            } graph_opt;

            //! get attribute for an operator
            inline const OprAttribute& get_opr_attribute(
                    OperatorNodeBase *opr) const;

            /*!
             * graph optimization level:
             * 0: disable
             * 1: level-1: inplace arith transformations during graph
             *    construction
             * 2: level-2: level-1, plus global optimization before graph
             *    compiling
             * 3: also enable JIT
             * <0: corresponding level, with result check for debug
             */
            int16_t graph_opt_level = 2;

            /*!
             * set logging level, larger number means more verbose
             * 0: no log info
             * 1: static memory allocation status
             *    WorkspaceLimitGetter summary
             *    optimizer summary
             * 2. optimizer var replace details during graph compiling
             *    duplicated operator
             */
            uint16_t log_level = 1;

            /*!
             * async exec: dispatch on separate threads for different comp_node
             * 0: do not perform async dispatch
             * 1: dispatch async if there are more than one comp node with
             *    limited queue
             * mask 0b10: async if there are multiple comp nodes with
             * mask 0b100: always async
             */
            uint16_t async_exec_level = 1;

            //! force dynamic memory alloc for all vars
            bool force_dynamic_alloc = false;

            //! whether to perform var sanity check on first run
            bool var_sanity_check_first_run = true;

            //! whether to allocate static memory just after compiling graph
            bool allocate_static_mem_after_graph_compile = false;

            /*!
             * whether only to perform non-computing tasks (like memory
             * allocation and queue initialization) for next exec. This would be
             * reset to false when the graph is executed.
             */
            bool fake_next_exec = false;

            //! whether to enable sublinear memory optimization
            bool enable_sublinear_memory_opt = false;

            //! Control parameter for sublinear memory optimization
            struct SublinearMemConfig {
                int thresh_nr_try = 10;
                int genetic_nr_iter = 0;
                int genetic_pool_size = 20;
                int lb_memory = 0;
                int num_worker = sys::get_cpu_count() / 2;
            } sublinear_mem_cofig;

            //! do not re-profile to select best impl algo when input shape
            //! changes (use previous algo)
            bool no_profiling_on_shape_change = false;

            //! whether to perform defragmenting when memory allocation for a
            //! dynamic var fails
            bool enable_var_mem_defragment = true;

            //! whether to reshape grad var whose wrt shape is statically
            //! inferrable but its own shape is dynamic
            bool enable_grad_var_static_reshape = false;

            /*!
             * whether to enable swap memory
             * as swap's performance is greatly worse than sublinear,
             * it is recommended to use sublinear first
             */
            bool enable_memory_swap = false;

            /*!
             * whether to use CompNodeSeqRecorder to record the execution
             * sequence and directly replay it for later executions.
             *
             * Level 1 is mainly used to speed up execution (especially for
             * opencl); level 2 is used for reducing memory usage.
             *
             * Level 1 constraints:
             *  1. All vars must be statically allocated
             *  2. Host input/output buffer pointers can not be changed if shape
             *     is not changed (this is not checked in execution for
             *     efficiency considerations; this is potentially dangerous)
             *  3. Synchronization can only occur at the end of execution
             *  4. Not all comp node implementations support recording computing
             *     sequence
             *  5. Only one comp node can be used in the graph
             *
             * Level 2: besides recording the computing sequence, the
             * dependencies are also moved into the compiled func (see
             * GraphExecutable::ExecDependency). Additional constraints:
             *  1. Shapes can not change
             *  2. both fake_next_exec and var_sanity_check_first_run must be
             *     disabled
             *  3. Var shapes must be correctly setup before calling compile()
             */
            uint8_t comp_node_seq_record_level = 0;

#if !MGB_BUILD_SLIM_SERVING
            //! whether to evaulate var node values as they are inserted
            bool eager_evaluation = false;
#endif

            //! add extra deps for the comp seq if a specific var is dependent
            ThinHashMap<VarNode*, VarNodeArray> extra_vardeps;

            //! contains any user data associated with this graph
            UserDataContainer user_data;
        }; // Options

        Options& options() {
            return m_options;
        }

        const Options& options() const {
            return m_options;
        }

        /*!
         * \brief get an instance for static var value infer manager
         */
        virtual static_infer::StaticInferManager& static_infer_manager() = 0;

        /*!
         * \brief get an instance for sequence computing node optimizer
         */
        virtual SeqCompNodeOptimizer& seq_comp_node_optimizer() = 0;

        /*!
         * \brief share static device memory with another computing graph
         *
         * To share memory for all graphs g[0..n-1], the correct way is to call
         * g[i].share_device_memory_with(g[0]) for i in range(1, n).
         *
         * This method must be called before compiling, and the user must ensure
         * AsyncExecutable objects with shared static device memory would not be
         * executed simultaneously.
         */
        virtual void share_device_memory_with(ComputingGraph &other) = 0;

        /*!
         * \brief set a custom DeviceMemoryAllocator to be used
         *
         * The given allocator would be used allocation in all graphs involved
         * in share_device_memory_with() calls related to this graph.
         */
        virtual void set_device_memory_allocator(
                std::shared_ptr<DeviceMemoryAllocator> allocator) = 0;

        /*!
         * \brief get size of currently allocated static device memory buffer on
         *      given computing node
         * \return memory size in bytes
         */
        virtual size_t get_device_memory_size(CompNode cn) = 0;

        /*!
         * \brief clear statically allocated device memory
         * \return use count of device memory before clear; a value of 1
         *      indicates the memory would be actually released
         */
        virtual size_t clear_device_memory() = 0;

        /*!
         * \brief set this graph as subgraph of another
         *
         * This mechanism is used to implement special control operators like
         * loop. Being a subgraph has following consequences:
         *   1. node ID counter would be shared
         *   2. when an AsyncExecutable compiled from subgraph are called, it
         *      would not wait for previous run to finish; instead, when
         *      AsyncExecutable from parent graph is being waited, it would call
         *      wait() on AsyncExecutables from the subgraph.
         *   3. some options would be passed from parent graph to sub graph
         *
         * Note that reference to subgraph should be kept by its owner
         * operator, whose reference is kept by parent graph.
         */
        virtual void set_as_subgraph(ComputingGraph &par_graph) = 0;

        //! get number of operators inserted in this graph
        virtual size_t nr_oprs_in_graph() const = 0;

#if !MGB_THREAD_SAFE
        /*!
         * \brief pre-allocate static storage used for internal states of
         *      computing graphs
         *
         * This is mainly used to reduce memory usage in single-threaded
         * environments. If a newly compiled function requires larger memory
         * size than previous ones, megbrain has to re-allocate static storage
         * buffer and the previous buffers are all wasted (because they should
         * have been shared with the largest buffer).
         *
         * If we know the max buffer size for all functions, the buffer can be
         * pre-allocated so it can be shared by all.
         *
         * A common practice to call prealloc_static_storage(0) to get the
         * current buffer size at the end of the program, and use this value as
         * the buffer size in next run.
         *
         * \param size anticipated max size of all buffers, in bytes
         * \return current buffer size
         */
        static size_t prealloc_static_storage(size_t size);
#endif

        /*!
         * \brief record given async error; it should call this function
         * rather than throw exception directly for the errors occurred
         * during calculation.
         */
        virtual void record_async_error(
                std::unique_ptr<MegBrainError> async_exc) = 0;

    private:
        SyncEventConnecter m_event;
        Options m_options;
        size_t m_id;
};

struct ComputingGraph::VarReceiverInfo {
    //! number of requests for directly computing by passing an empty callback
    size_t nr_direct_comp_req = 0;

    //! number of operators that need device value of this var
    size_t dev_value = 0;

    //! last dev value reader in the computing sequence
    OperatorNodeBase* last_dev_value_reader = nullptr;

    //! number of operators that need shape of this var, which can not be
    //! statically inferred
    size_t shape = 0;

    //! number of operators that need host value of this var, which can not be
    //! statically inferred
    size_t host_value = 0;

    //! number of operators in \p dev_value and \p host_value that allow this
    //! var to be empty
    size_t allow_empty_value = 0;

    //! whether nothing is needed completely
    bool empty() const {
        return !nr_direct_comp_req && !dev_value && !shape && !host_value;
    }

    //! whether computing value is needed (i.e. either dev_value, or shape, or
    //! host_value)
    bool value_needed() const {
        return dev_value || shape || host_value;
    }

    //! whether this var can be empty
    bool is_empty_allowed() const {
        return allow_empty_value == host_value + dev_value;
    }

    std::string to_string() const;
};

/*!
 * \brief helper function for creating an operator with unique output and
 *      inserting it into graph
 */
template<typename Node, typename ...Args>
SymbolVar SymbolVar::insert_single_output_opr(Args &&...args) const {
    return m_node->owner_graph()->insert_opr(
            std::make_unique<Node>(std::forward<Args>(args)...))->output(0);
}

} // namespace cg
} // namespace mgb


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
