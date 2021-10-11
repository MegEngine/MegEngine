/**
 * \file src/core/include/megbrain/graph/var_node.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/bases.h"
#include "megbrain/utils/comp_node_sync_manager.h"
#include "megbrain/utils/mempool.h"
#include "megbrain/utils/small_vector.h"

#include <atomic>
#include <mutex>
#include "megbrain/tensor.h"

namespace mgb {
namespace imperative {
class ProxyGraph;
namespace proxy_graph {
class ProxyGraph;
}
}  // namespace imperative

namespace cg {
namespace static_infer {
class StaticInferManagerImpl;
}

class VarDevMemDefragmenter;
class EagerEvalManager;

/*!
 * \brief memory allocation plan held by a variable
 *
 * A MemAllocPlan is a view (i.e. with offset and layout) for some Chunk; Memory
 * sharing between vars is implemented by sharing a Chunk of their mem plans.
 */
class MemAllocPlan final : public json::Serializable, public NonCopyableObj {
public:
    /*!
     * \brief identifier for allocated memory
     *
     * Each Chunk object corresponds to an allocated memory chunk. Memory
     * forwarding and force updating are implemented by sharing Chunk
     * objects between vars.
     *
     * If mem_alloc_status is not invalid, the memory region for this chunk
     * is owner_var->dev_tensor().storage().
     */
    class Chunk : public NonCopyableObj {
        friend class MemAllocPlan;
        friend class VarDevMemDefragmenter;

        std::atomic_size_t m_refcnt;
        size_t m_size;

    public:
        /*!
         * \brief memory allocation status for this chunk
         *
         * Allocation status can either be INVALID, FROM_OWNER_VAR, or an
         * offset in a static allocation buffer. This status is compactly
         * represented by an integer value. No error check is performed in
         * the accessors.
         *
         * Note that for static_offset, it is set in
         * SeqMemOptimizer::plan_chunk_allocation() and accessed in
         * VarNodeMemManager::make_static_var_tensor_from_alloc_plan()
         */
        class MemAllocStatus {
            static constexpr size_t INVALID = 0, FROM_OWNER_VAR = 1, OFFSET = 2;
            size_t m_val = INVALID;

        public:
            //! whether memory is not allocated yet
            bool is_invalid() const { return m_val == INVALID; }

            //! whether memory comes from owner_var->dev_tensor()
            bool is_from_owner_var() const { return m_val == FROM_OWNER_VAR; }

            //! whether memory is statically allocated
            bool is_static_offset() const { return m_val >= OFFSET; }

            size_t static_offset() const { return m_val - OFFSET; }

            void set_invalid() { m_val = INVALID; }

            void set_from_owner_var() { m_val = FROM_OWNER_VAR; }

            void set_static_offset(size_t offset) { m_val = offset + OFFSET; }
        };

        //! var that first creates this chunk
        VarNode* const owner_var;

        MemAllocStatus mem_alloc_status;

        //! size of this chunk in bytes
        size_t size() const { return m_size; }

        //! update value of m_size, only used in dynamic var allocation
        void update_size_for_dynamic_alloc(size_t size) { m_size = size; }

        std::string id_str() const;

#if MGB_ENABLE_JSON
        std::shared_ptr<json::Value> to_json() const;
#endif

        explicit Chunk(VarNode* ov) : owner_var(ov) {}
    };

    explicit MemAllocPlan(VarNode* owner_var);

    bool valid() const { return m_chunk; }

    //! dtype of owner var
    inline DType dtype() const;

    //! tensor format of owner var
    inline TensorFormat format() const;

    //! get associated chunk
    Chunk& chunk() {
        mgb_assert(valid());
        return *m_chunk;
    }

    bool is_invalid_cond_exec() const {
        return m_chunk == &sm_chunk_invalid_cond_exec_marker;
    }

    //! get offset in bytes of this MemAllocPlan in associated chunk
    size_t offset_in_chunk_byte() const { return m_offset_byte; }

    const TensorLayout& layout() const { return m_layout; }

    MemAllocPlan& layout(const TensorLayout& dest, bool allow_shape_change = false);

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const override;
#endif

    /*!
     * \brief release current chunk and decr its refcnt
     *
     * Release tensor storage if refcnt drops to zero
     */
    MemAllocPlan& release_chunk();

    /*!
     * \brief reset chunk to a privately owned chunk, and setup offset and
     *      layout from owner var, and clear tensor storage
     */
    MemAllocPlan& reset_from_owner_var();

    /*!
     * \brief reset to a special marker that indicates this var is not
     *      computed in conditional execution
     *
     * This is used in VarNodeMemManager to detect if the var is invalid
     * without adding a new field.
     */
    MemAllocPlan& reset_as_invalid_cond_exec() {
        m_chunk = &sm_chunk_invalid_cond_exec_marker;
        return *this;
    }

    /*!
     * \brief reset to uninitialized status
     *
     * This is called before calling OperatorNodeBase::init_output_mem_plan
     * and before memplan optimization.
     */
    MemAllocPlan& reset_to_uninitialized() {
        m_layout.ndim = 0;
        m_chunk = nullptr;
        m_readonly_fwd_list.reset();
        return *this;
    }

    //! assign layout, offset and chunk from another mem alloc plan
    MemAllocPlan& assign(const MemAllocPlan& src);

    //! assign for readonly forward
    MemAllocPlan& assign_for_forward(const MemAllocPlan& src, const SubTensorSpec& sub);

    /*!
     * \brief next readonly-forward reader of this MemAllocPlan
     *
     * All the readers of a MemAllocPlan form a singly-linked list which is
     * maintained by assign_for_forward().
     */
    MemAllocPlan* next_readonly_fwd_reader() const {
        return m_readonly_fwd_list.next();
    }

    //! the var that owns this mem plan
    VarNode* owner_var() const { return m_chunk_storage.owner_var; }

private:
    class ReadonlyFwdList {
        MemAllocPlan *m_prev = nullptr, *m_next = nullptr;
        static MGB_MUTEX list_mutex;

    public:
        MemAllocPlan* next() const { return m_next; }
        void reset();
        inline void insert_after(const MemAllocPlan& prev, MemAllocPlan* self);
        inline void remove_self();
    };

    static Chunk sm_chunk_invalid_cond_exec_marker;

    TensorLayout m_layout;      //!< actual layout; shape must equal to var shape
    size_t m_offset_byte = -1;  //!< offset in m_chunk
    Chunk* m_chunk = nullptr;
    Chunk m_chunk_storage;
    mutable ReadonlyFwdList m_readonly_fwd_list;
};

class VarNodeMemManager;

/*!
 * \brief Node for a variable.
 *
 * It must be the output of exactly one OperatorNode and may be input to other
 * OperatorNode.
 *
 * Each variable has an owner, the operator that generates this variable as one
 * of the output.
 *
 * VarNode class exposes most commonly used memory management interface
 */
class VarNode final : public GraphNodeBase {
public:
    /*!
     * \brief this constructor should only be called by
     *      OperatorNodeBase::add_output
     *
     * implemented in core/impl/graph/operator_node.cpp
     */
    inline VarNode(Maybe<std::string> name, OperatorNodeBase* owner);

    /* ===================== memory optimization ===================== */

    using LayoutConstraintCallback = thin_function<bool(const TensorLayout&)>;

    /*!
     * \brief add a callback function to check the validity of a particular
     *      tensor layout
     *
     * If callback returns true, it means that this VarNode's dev_tensor
     * with given layout may be forwarded to opr directly, otherwise it
     * will be implicitly rearranged to a contiguous one.
     */
    VarNode& add_layout_constraint(LayoutConstraintCallback callback);

    /*!
     * \brief requires the layout to be contiguous
     *
     * Note: since many oprs require inputs to be contiguous, this is
     * implemented by marking a flag on the var rather than adding a
     * LayoutConstraintCallback to check whether it is contiguous. All the
     * existing callbacks would be cleared and new callbacks would be
     * ignored after add_layout_constraint_contiguous() is invoked.
     */
    VarNode& add_layout_constraint_contiguous();

    /*!
     * \brief requires the layout to be monotone while allowing broadcast
     *
     * Note: similar to add_layout_constraint_contiguous() this is
     * implemented by marking a flag; however user-defined callbacks are
     * still invoked since they might impose stronger constraints.
     */
    VarNode& add_layout_constraint_monotone();

    /*!
     * \brief request that memory should be readonly forwarded from other
     *      var
     *
     * Note that this function must be called from
     *      OperatorNodeBase::mem_plan_fwd_in2out_readonly.
     *
     * \return whether this request could be satisfied
     */
    MGB_WARN_UNUSED_RESULT bool set_fwd_in2out_readonly(
            VarNode* input, const SubTensorSpec& sub);

    /*!
     * \brief request that this var share memory with another var, whose
     *      content would also be modified
     *
     * Note that this function must be called from
     *      OperatorNodeBase::mem_plan_fwd_in2out_writable.
     */
    VarNode& set_fwd_in2out_writable(VarNode* input);

    /*!
     * \brief require this var to share memory from another var; only used
     * for operators that have an explicit updating semantics
     *
     * Note that this function must be called during operator node
     * initialization
     */
    VarNode& set_fwd_in2out_writable_force(VarNode* input);

    /* ===================== getter and setters =====================  */

    OperatorNodeBase* owner_opr() const { return m_owner; }

    //! get name; if name is not valid, get name of owner opr
    const std::string& name() const;

    //! get name as C-string
    const char* cname() const { return name().c_str(); }

    //! whether name is explicitly set,
    bool has_name_set() const { return m_has_name_set; }

    //! set name explicitly
    VarNode& name(std::string name);

    //! get data type of data in this var
    DType dtype() const { return m_dev_tensor.dtype(); }

    //! get tensor format in this var
    TensorFormat format() const { return m_dev_tensor.format(); }

    //! set dtype; this function can only be called once
    VarNode& dtype(DType dtype);

    //! set format; this function can only be called once
    VarNode& format(TensorFormat format);

    MemAllocPlan& mem_plan() { return m_mem_plan; }

    bool dev_tensor_valid() const {
        return m_mem_plan.valid() && m_mem_plan.layout().eq_shape(m_shape) &&
               m_dev_tensor.storage().comp_node_valid() &&
               m_dev_tensor.layout().eq_layout(m_mem_plan.layout()) &&
               m_dev_tensor.comp_node() == m_comp_node;
    }

    //! get the underlying device tensor to fill data
    const DeviceTensorND& dev_tensor() const;

    /*!
     * \brief get the underlying device tensor that can be modified(like
     *      resize())
     *
     * This should only be called from the owner opr of this var, and this
     * var must have flag NO_SYS_MEM_ALLOC.
     */
    DeviceTensorND& mutable_dev_tensor();

    /*!
     * \brief previous dev ptr before deallocating dev_tensor; used for
     *      testing and debugging
     */
    const void* prev_dev_ptr() const { return m_prev_dev_ptr; }

    /*!
     * \brief get the comp node on which this var is computed
     */
    CompNode comp_node() const { return m_comp_node; }

    /*!
     * \brief set comp node; only the memory node could be changed if called
     *      multiple times
     */
    VarNode& comp_node(const CompNode& cn);

    const TensorShape& shape() const { return m_shape; }

    //! get current reference count; not thread safe, and only used for
    //! testing purposes
    size_t refcnt() const { return m_refcnt; }

    /*!
     * \brief reset VarNode shape
     * \return whether shape differs from old shape
     */
    VarNode& shape(const TensorShape& shape);

    bool allow_shape_change() const { return m_allow_shape_change; }

    const TensorLayout& layout() const {
        mgb_assert(m_mem_plan.valid() && m_mem_plan.layout().eq_shape(m_shape));
        return m_mem_plan.layout();
    }

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> to_json() const override;
#endif

    /*!
     * \brief add a callback to be executed when shape of this var is
     *      updated
     * \param tag callback tag; each tag can have at most one callback
     */
    void add_shape_update_callback(void* tag, thin_function<void(VarNode*)> cb) {
        m_shape_update_callback[tag] = cb;
    }

    enum class Flag : uint32_t;

    VarNode& add_flag(Flag flag);

    inline bool contain_flag(Flag flag) const;

    /* ===================== dynamic memory ===================== */

    /*!
     * \brief set shape and alloc memory storage
     *
     * This function should only be called by this var's owner operator and
     * this var must have NO_SYS_MEM_ALLOC flag; if shape does not increase
     * and original tensor storage is valid, it is guaranteed that old data
     * would be retained.
     *
     * \warning Alloc size_req memory if size_req != 0.
     */
    VarNode& shape_alloc(const TensorShape& shape, size_t size_req = 0);

    /*!
     * \brief directly reset device tensor from another var
     *
     * This function should only be called by this var's owner operator and
     * this var must have NO_SYS_MEM_ALLOC flag. It can be used to forward
     * var values in the same graph or between graph. If both \p src_var and
     * this var belong to same graph, memory forwarding may fail (e.g. when
     * \p src_var is force updated by another opr)
     *
     * \param src_var the var node to provide dev tensor, which must have
     *      been initialized, and does not have to be in the same computing
     *      graph. Its value must be contiguous or empty. It can also be
     *      placed on a different comp node.
     *
     * \return whether memory forwarding succeeds; if false is returned, a
     *      new tensor would be allocated and its value is copied from src
     */
    MGB_WARN_UNUSED_RESULT bool reset_dev_tensor_from_other_var(VarNode* src_var);

    /*!
     * \brief directly reset device tensor from a given tensor
     *
     * This function should only be called by this var's owner operator and
     * this var must have NO_SYS_MEM_ALLOC flag
     *
     * \param value the tensor to be used; it must be contiguous or empty
     *       and be placed on the same comp node of this var.
     */
    VarNode& reset_dev_tensor_from_tensor(const DeviceTensorND& value);

    /*!
     * \brief add a var to add RT_FORCE_DYNAMIC_MEM_ALLOC flag if such flag
     *      is added to this var
     *
     * The chains form a directed graph, and when a var is added
     * RT_FORCE_DYNAMIC_MEM_ALLOC by VarNodeMemManager, all nodes in the
     * connected component would be added with such flag.
     *
     * This method should be called from
     * OperatorNodeBase::init_rt_force_dynamic_mem_alloc_imply_chain impls.
     */
    VarNode& add_rt_force_dynamic_mem_alloc_imply_chain(VarNode* dest);

    /* ===================== graph compiler special ===================== */

    /*!
     * \brief initialize mem plan as a uniquely owned contiguous chunk
     *
     * this function should only be called from
     * OperatorNodeBase::init_output_mem_plan and shape and comp_node must
     * have been setup.
     *
     * \param fixed_alloc if not null, it should be a tensor providing
     *      memory allocation for this var.
     */
    MemAllocPlan& init_mem_plan(const DeviceTensorND* fixed_alloc = nullptr);

    /*!
     * \brief get the shape and value infer trait
     */
    const std::tuple<void*, void*>& get_static_infer_trait() {
        return m_static_infer_trait;
    }

private:
    //! whether its memory should be allocated by mgb system during graph
    //! execution; initialized in VarNodeMemManager::reset_opr_seq()
    bool m_should_sys_alloc = false;
    bool m_has_name_set = false;
    //! whether to allow shape being modified; used by eager const shape in
    //! static infer
    bool m_allow_shape_change = true;
    Maybe<std::string> m_name;
    OperatorNodeBase* const m_owner;

    const void* m_prev_dev_ptr = nullptr;
    Flag m_flag = static_cast<Flag>(0);
    TensorShape m_shape;

    CompNode m_comp_node;
    DeviceTensorND m_dev_tensor;
    MemAllocPlan m_mem_plan{this};
    ThinHashMap<void*, thin_function<void(VarNode*)>> m_shape_update_callback;
    //! synchronizer that is managed by SeqCompNodeOptimizer
    CompNodeSyncManager* m_cn_sync_manager = nullptr;

    /*!
     * used by StaticInferManagerImpl to store the static infer trait
     * associated with this var.
     *
     * Almost every VarNode has an associated TagTraitContainer, so its
     * storage is inlined into VarNode.
     */
    std::tuple<void*, void*> m_static_infer_trait{nullptr, nullptr};

    /*!
     * number of readers that rely on value of m_dev_tensor, used for
     * dynamic memory management. m_refcnt is initialized as m_refcnt_init.
     * For statically allocated vars and NO_MEM_RECLAIM vars, m_refcnt_init
     * is set to inf; otherwise it is the total number of outputs of reader
     * oprs that has DEV_VALUE dep on this var. After completion of each opr
     * on each comp node, or completion of a callback, m_refcnt would be
     * decreased; if it reaches zero, m_dev_tensor and m_mem_plan would be
     * released.
     */
    std::atomic_size_t m_refcnt{0};
    size_t m_refcnt_init = 0;

    std::vector<VarNode*> m_rt_force_dynamic_mem_alloc_imply_chain;

    void modify_flag(Flag delta, Flag new_flag);

    void assign_dev_tensor_from_tensor(const DeviceTensorND& value);

#if MGB_ENABLE_JSON
    std::shared_ptr<json::Value> dump_static_infer_info_to_json() const;
#endif

    friend class static_infer::StaticInferManagerImpl;
    friend class VarNodeMemManager;
    friend class VarDevMemDefragmenter;
    friend class EagerEvalManager;
    friend class MemAllocPlan;
    friend class imperative::ProxyGraph;
    friend class imperative::proxy_graph::ProxyGraph;
};

enum class VarNode::Flag : uint32_t {
    //! do not allocate memory by the system allocator even if shape could be
    //! inferred
    NO_SYS_MEM_ALLOC = 1 << 0,

    //! do not allocate memory if value of this var is not used (i.e.
    //! VarReceiverInfo::value_used() returns false)
    NO_ALLOC_IF_UNUSED = 1 << 1,

    /*!
     * do not allocate memory statically (would be allocated dynamically if
     * possible); useful if a var in subgraph would be directly forwarded to a
     * var in owner graph (e.g.  in case for LAST output mode in Loop)
     */
    NO_SYS_STATIC_MEM_ALLOC = 1 << 2,

    /*!
     * do not reclaim memory
     * if NO_SYS_MEM_ALLOC is set or this var has dynamic storage, memory would
     *      not be reclaimed after all readers are processed
     * if this var has satic storage, its memory would not be reused by others
     */
    NO_MEM_RECLAIM = 1 << 3,

    /*!
     * var node used as temporary storage, whose content should
     * not be read by others
     */
    VOLATILE_CONTENT = 1 << 4,

    /*!
     * allow this var to have empty shape, which means it would not consume any
     * memory and it has nullptr as the underlying pointer; vars without this
     * flag set would trigger an error during memory allocation to avoid
     * uninitialized output var shape. This flag should be set by the owner opr.
     */
    ALLOW_EMPTY_SHAPE = 1 << 5,

    /*!
     * value is always available on device even before opr is executed (e.g.
     * SharedDeviceTensor), so various optimizations can be performed
     */
    PERSISTENT_DEVICE_VALUE = 1 << 6,

    /*!
     * disallow RT_FORCE_DYNAMIC_MEM_ALLOC added to this node during memory
     * optimization; this is only applicable when the operator manages memory
     * of this var manually, and the memory is never reclaimed. Must be used
     * with NO_MEM_RECLAIM.
     */
    DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC = 1 << 7,

    /*!
     * disable sanity check for this VarNode
     * this flag is added for swap_memory; SwapInMS opr works as a trigger to
     * make its output VarNode start copying from host parallelly, when
     * SwapInMS finishs execute(), it is likely that its output tensor does not
     * have 'exact' content, so we need to disable var_sanity_check in this case
     */
    DISALLOW_VAR_SANITY_CHECK = 1 << 8,

    /*!
     * force dynamic memory allocation even if shape could be statically
     * inferred; conflicts with NO_SYS_MEM_ALLOC
     *
     * note that this is a runtime-flag, which would be cleared and re-evaluated
     * on graph compiling; it is set up by VarNodeMemManager and propagated
     * through
     */
    RT_FORCE_DYNAMIC_MEM_ALLOC = 1 << 9,

    /*!
     * this flag indicates that the opr has been inserted into the graph and
     * certain flags can not be modified. Only NO_MEM_RECLAIM,
     * NO_SYS_STATIC_MEM_ALLOC and RT_FORCE_DYNAMIC_MEM_ALLOC flags can be added
     * after FLAG_FREEZED is present.
     */
    FLAG_FREEZED = 1 << 10,

    /*!
     * this flag indicates that data of this var has been processed and no need
     * later, it can be freed, this is used in weight preprocess for memory save
     */
    MEMORY_NO_NEED = 1 << 11,
};

MGB_DEF_ENUM_CLASS_BIT_OPR(VarNode::Flag)

bool VarNode::contain_flag(Flag flag) const {
    return static_cast<bool>(m_flag & flag);
}

using VarNodeSet = ThinHashSet<VarNode*>;

DType MemAllocPlan::dtype() const {
    return m_chunk_storage.owner_var->dtype();
}

TensorFormat MemAllocPlan::format() const {
    return m_chunk_storage.owner_var->format();
}

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
