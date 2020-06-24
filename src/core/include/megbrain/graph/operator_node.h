/**
 * \file src/core/include/megbrain/graph/operator_node.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/static_infer.h"
#include "megbrain/graph/var_node.h"
#include "megbrain/graph/symbol_var.h"

#include "megbrain/utils/hashable.h"
#include "megbrain/utils/enum_class_bit.h"
#include "megbrain/utils/thin/hash_table.h"
#include "megbrain/utils/small_vector.h"

#include <type_traits>

namespace mgb {
namespace cg {

class ExecutionMask;

/*!
 * \brief configuration for operator nodes
 */
class OperatorNodeConfig final: public Hashable {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    public:
        using CompNodeArray = SmallVector<CompNode, 1>;

        OperatorNodeConfig() = default;
        ~OperatorNodeConfig();

        OperatorNodeConfig(std::string name):
            m_name{std::move(name)}
        {}

        OperatorNodeConfig(const CompNode &cn) {
            comp_node(cn);
        }

        OperatorNodeConfig(std::string name, const CompNode& cn,
                           DType dtype = {})
                : m_name{std::move(name)}, m_output_dtype{dtype} {
            comp_node(cn);
        }

        explicit OperatorNodeConfig(DType dtype) : m_output_dtype{dtype} {};

        /*!
         * \brief make a name according to default name and input vars
         */
        std::string make_name(std::string default_name,
                const VarNodeArrayView& input_var, size_t opr_id) const;

        /*!
         * \brief set node name
         */
        OperatorNodeConfig& name(std::string name) {
            m_name = std::move(name);
            return *this;
        }

        const Maybe<std::string>& name() const {
            return m_name;
        }

        /*!
         * \brief update instance ID
         *
         * Instance ID is a hashed value used to differentiate multiple
         * instances of the same operator (with same inputs, params and
         * config), so the deduplication system can be bypassed.
         *
         * This method always updates underlying instance_id.
         */
        template<typename T>
        OperatorNodeConfig& update_instance_id(const T& p) {
            static_assert(std::is_pointer<T>::value,
                "update_instance_id can only accept a pointer");
            m_instance_id_hashed = hash_pair_combine(
                m_instance_id_hashed, mgb::hash(p));
            return *this;
        }

        /*!
         * \brief reset instance ID to the initial value
         */
        OperatorNodeConfig& reset_instance_id() {
            m_instance_id_hashed = sm_initial_instance_id;
            return *this;
        }

        /*!
         * \brief get current hashed instance ID
         */
        size_t instance_id() const {
            return m_instance_id_hashed;
        }

        /*!
         * \brief set preferred single comp node
         */
        OperatorNodeConfig& comp_node(const CompNode &node);

        /*!
         * \brief directly set all the CompNodes
         */
        OperatorNodeConfig& comp_node_arr(const CompNodeArray &arr);

        /*!
         * \brief get single comp node if the user has set it, or an invalid
         *      comp node if the config is empty
         */
        CompNode get_single_comp_node() const;

        /*!
         * \brief follow the computing node of dest
         */
        OperatorNodeConfig& follow_comp_node(const SymbolVar &dest) {
            return comp_node(dest.node()->comp_node());
        }

        OperatorNodeConfig& output_dtype(DType dtype);

        DType output_dtype() const { return m_output_dtype; }

        /*!
         * \brief whether at least one comp node has been set
         */
        bool has_comp_node_set() const {
            return !m_comp_node.empty();
        }

        const CompNodeArray& comp_node() const {
            return m_comp_node;
        }

        size_t hash() const override;

        bool is_same_st(const Hashable &rhs) const override;

    private:
        static constexpr size_t sm_initial_instance_id = 1333331;
        Maybe<std::string> m_name;
        CompNodeArray m_comp_node;
        size_t m_instance_id_hashed = sm_initial_instance_id;
        DType m_output_dtype;
};


/*!
 * \brief executable used internally for cg
 */
class GraphExecutable {
public:
    class ExecEnv;
    class ExecDependency;

    using ExecDependencyArray = std::vector<std::unique_ptr<ExecDependency>>;

    virtual void execute(ExecEnv& env) = 0;

    /*!
     * \brief append the dependencies into \p record
     *
     * Some deps might be moved; the original operator should not be used again.
     *
     * The default implementation does nothing
     */
    virtual void record_execute_deps(ExecDependencyArray& record);

protected:
    ~GraphExecutable() = default;
};

/*!
 * \brief dependency for execute()
 *
 * This is used when comp_node_seq_recorder_level is 2: the objects needed by
 * kernels in execute() should be moved to this object and the parent operator
 * class would later be destructed.
 */
class GraphExecutable::ExecDependency {
public:
    virtual ~ExecDependency() = default;

    //! if this returns true, do_runtime_check() would be called before each run
    virtual bool has_runtime_check() const;

    virtual void do_runtime_check();
};


/*!
 * \brief operator execution environment
 *
 * When GraphExecutable::execute() is called, it should add tasks into the
 * ExecEnv. The tasks added to ExecEnv would be invoked by a scheduler to
 * perform actual computing.
 *
 * Operator code usually only calls dispatch_on_comp_node()
 *
 * Note: The ExecEnv class exists as an abstraction layer for controlling
 * asynchronous kernel dispatching behavior. CUDA has a limited task queue so we
 * have to use a dedicated CPU thread for each CUDA stream (it can be treated as
 * a blocking queue). But for CPU we have our own unlimited dispatch queue so
 * the ExecEnv can be synchonous.
 */
class GraphExecutable::ExecEnv {
protected:
    ~ExecEnv() noexcept = default;

public:
    using Task = thin_function<void()>;

    //! add a task to the queue corresponding to given comp node
    virtual void dispatch_on_comp_node(CompNode cn, Task&& task) = 0;

    //! like dispatch_on_comp_node, but with specific mask other than current
    //! opr mask
    virtual void dispatch_on_comp_node_with_mask(CompNode cn, Task&& task,
                                                 ExecutionMask* mask) = 0;

    /*!
     * \brief pause execution on all threads if there are async dispatch
     *      threads
     *
     * This is currently only used by memory defragmenter.
     */
    virtual void pause_exec() = 0;

    /*!
     * \brief resume execution (cancel previous pause_exec())
     */
    virtual void resume_exec() = 0;
};

/*!
 * \brief properties of an operator
 *
 * Most of the fields are setup by OperatorNodeBase::do_make_node_prop() and
 * can not be changed later; but attribute() can always be modified.
 */
class OperatorNodeProp final : public NonCopyableObj {
public:
    enum class Flag : uint32_t {
        /*!
         * the opr works on a single comp node
         */
        SINGLE_COMP_NODE = 1 << 0,

        /*!
         * the opr could work on different memory node than its input
         */
        CROSS_COMP_NODE_MEMORY = 1 << 1,

        /*!
         * not a pure function meaning output is not completely determined by
         * input; also means that multiple evaluation of the same (operator
         * without returning control to user) may produce different results
         */
        IMPURE_FUNC = 1 << 2,

        /*!
         * content of input var would be modified (currently only AddUpdate)
         */
        FORCE_UPDATE_INPUT_VAR = 1 << 3,

        /*!
         * do not allow comp node optimizer to change comp node of output vars
         * of this operator
         */
        DISALLOW_COMP_NODE_OPTIMIZE = 1 << 4,

        /*!
         * the operator should not be automatically duplicated (i.e. it may have
         * side effect, even if it is a pure function); automatic duplication
         * can be used in sublinear memory optimizer
         */
        NO_AUTOMATIC_DUP = 1 << 5,

        /*!
         * this operator has custom implementation of init_output_mem_plan and
         * it may change even if no shape changes. init_output_mem_plan() for
         * those oprs would always be called before each graph execution.
         */
        IMPURE_OUTPUT_MEM_PLAN = 1 << 6,

        /*!
         * Do not automatically add waiting spec for inputs on output comp
         * nodes. This is useful for utility operators that directly dispatch
         * funcs onto input comp nodes; their outputs are usually a placeholder
         * variable.
         *
         * Note: the input_waiting_spec() would not be initialized and the
         * output should not be read by oprs on other comp nodes;
         */
        NO_INPUT_WAITING = 1 << 7,
    };

    //! operator attributs that can be directly modified
    struct Attribute {
        //! objects associated with this opr; their memory should be managed by
        //! some UserData class attached to the computing graph
        class Accessory {
            MGB_IF_COND_EXEC(friend class ExecutionMask);
            MGB_IF_COND_EXEC(ExecutionMask* exec_mask = nullptr);
        };

        //! source operator that creates this opr as its gradient
        struct GradTracker {
            OperatorNodeBase* orig_opr;
            VarNode *target_var, *wrt_var;
        };

        //! topo sort priority: smaller number means higher priority
        int priority = 0;

        Accessory accessory;

        Maybe<GradTracker> grad_tracker;

        /*!
         * if this operator is copied from another opr or generated by graph
         * transformation from another opr, then \p src_opr would be the
         * corresponding source operator
         */
        OperatorNodeBase* src_opr = nullptr;
    };

    /*!
     * \brief type of dependency of one operator on another operator
     */
    enum class DepType : uint32_t {
        /*!
         * device value must be computed before starting opr; this is the
         * default dep type for input vars
         */
        DEV_VALUE = 1 << 0,

        /*!
         * depends on host value, which must be retrieved from
         * StaticInferManager during runtime; if value could be statically
         * inferred and DEV_COMP_ORDER is not set, it may not be computed on
         * device; note that change of host value would not cause memory
         * reallocation, so oprs whose memory depends on host value but output
         * shape may be unchanged should add HOST_VALUE_DYNOUT
         */
        HOST_VALUE = 1 << 1,

        /*!
         * add RT_FORCE_DYNAMIC_MEM_ALLOC flag to output if input in this
         * dependency entry is not const-inferable. HOST_VALUE must also be set.
         *
         * This is used when output value can be forwarded from one input (e.g.
         * value in IndexAt opr) and other inputs (e.g. index in IndexAt) change
         * frequently. Also note that static memory allocation would not be
         * triggered when no shape changes. So oprs like IndexAt must use
         * dynamic allocation to ensure its output value corresponds to current
         * index value if index can change.
         */
        HOST_VALUE_DYNOUT = 1 << 2,

        /*!
         * depends on shape, which can be accessed by VarNode::shape during
         * runtime; if shape could be statically inferred and DEV_COMP_ORDER is
         * not set, computing on device may be omitted
         */
        SHAPE = 1 << 3,

        /*!
         * only needs to ensure it has been computed; Note that value is not
         * needed so memory could be reclaimed, but shape is always valid
         */
        DEV_COMP_ORDER = 1 << 4,

        /*!
         * whether empty tensor is allowed for HOST_VALUE or DEV_VALUE dep
         * types; either HOST_VALUE or DEV_VALUE must also be specified
         */
        VALUE_ALLOW_EMPTY = 1 << 5,
    };

    using DepMap = ThinHashMap<VarNode*, DepType>;

    /*!
     * \brief get all dependency needed to produce output
     */
    const DepMap& dep_map() const { return m_dep_map; }

    DepMap& dep_map() { return m_dep_map; }

    /*!
     * \brief add a flag
     */
    inline OperatorNodeProp& add_flag(Flag flag);

    /*!
     * \brief test whether a flag has been added
     */
    inline bool contain(Flag req) const;

    /*!
     * \brief add dependency type to a var; original dependency types would
     *      be retained; \p dest is allowed to not exist in current dep map
     */
    inline OperatorNodeProp& add_dep_type(VarNode* dest, DepType type);

    //! like add_dep_type() but requires \p dest to already exist in dep map
    inline OperatorNodeProp& add_dep_type_existing_var(VarNode* dest,
                                                       DepType type);

    /*!
     * \brief reset dep type; the vars could contain duplicated var nodes,
     *      in which case the corresponding dep type would be ORed together
     */
    void reset_dep_type(const VarNodeArray& vars,
                        const SmallVector<DepType>& dep_types);

    /*!
     * \brief whether a dep type require device computation order
     */
    static inline constexpr bool is_device_comp_order_dep(DepType type);

    /*!
     * \brief whether a dep type require values on device
     */
    static inline constexpr bool is_device_value_dep(DepType type);

    //! user-modifiable attribute
    Attribute& attribute() const { return m_attribute; }

private:
    friend class OperatorNodeBase;

    Flag m_flag = static_cast<Flag>(0);
    DepMap m_dep_map;
    mutable Attribute m_attribute;
};

MGB_DEF_ENUM_CLASS_BIT_OPR(OperatorNodeProp::Flag)
MGB_DEF_ENUM_CLASS_BIT_OPR(OperatorNodeProp::DepType)

constexpr bool OperatorNodeProp::is_device_comp_order_dep(DepType type) {
    return static_cast<bool>(type &
                             (DepType::DEV_VALUE | DepType::DEV_COMP_ORDER));
}

OperatorNodeProp& OperatorNodeProp::add_dep_type(VarNode* dest, DepType type) {
    DepType& v = m_dep_map[dest];
    v = v | type;
    return *this;
}

OperatorNodeProp& OperatorNodeProp::add_dep_type_existing_var(VarNode* dest,
                                                              DepType type) {
    DepType& v = m_dep_map.at(dest);
    v = v | type;
    return *this;
}

constexpr bool OperatorNodeProp::is_device_value_dep(DepType type) {
    return static_cast<bool>(type & DepType::DEV_VALUE);
}

OperatorNodeProp& OperatorNodeProp::add_flag(Flag flag) {
    m_flag = m_flag | flag;
    return *this;
}

bool OperatorNodeProp::contain(Flag req) const {
    return static_cast<bool>(m_flag & req);
}

/*!
 * \brief Node for an operator.
 *
 * An operator is defined to be a node that could generate one or more VarNode
 * as output.
 *
 * Each operator node must be purely functional, i.e. the same node evaluated on
 * the same input value must produce the same output value
 *
 * Each operator has an owner, the computing graph that it belongs to
 */
class OperatorNodeBase: public GraphNodeBase, public Hashable,
                        public GraphExecutable {
    public:
        using NodeProp = OperatorNodeProp;

        //! pack of params in constructor, to ease inheritance
        struct CtorParamPack {
            ComputingGraph *owner;
            const OperatorNodeConfig &config;
            const std::string &default_name;
            const VarNodeArrayView &input_var_naming;
        };

        virtual ~OperatorNodeBase() noexcept;

#if MGB_ENABLE_JSON
        /* ===================== json io ===================== */
        std::shared_ptr<json::Value> to_json() const override;

        //! extra value to be added to json
        std::shared_ptr<json::Object> to_json_extra_json = json::Object::make();
#endif

        /* ===================== misc getters/setters ===================== */

        const std::string& name() const { return m_name; }

        const char* cname() const { return m_name.c_str(); }

        void name(std::string name) { m_name = std::move(name); }

        const VarNodeArray& input() const { return m_input; }

        const VarNodeArray& output() const { return m_output; }

        // non-volatile outputs
        const VarNodeArray usable_output() const;

        VarNode* input(size_t idx) const { return m_input.at(idx); }

        VarNode* output(size_t idx) const { return m_output.at(idx); }

        //! hash that combines all inputs, m_config.comp_node() and all
        //! add_equivalence_component calls
        size_t hash() const override final;

        /*!
         * \brief get node prop, which is available and constant after node
         *      construction
         *
         * Note that this function calls do_make_node_prop() on first call
         */
        const NodeProp& node_prop() const;

        /*!
         * \brief called by ComputingGraph to mark that this node has been
         *      inserted in graph; inputs and outputs could not be later changed
         */
        void set_inserted_in_graph() { m_inserted_in_graph = true; }

        bool inserted_in_graph() const { return m_inserted_in_graph; }

        const OperatorNodeConfig& config() const { return m_config; }

        /* ===================== execution ===================== */

        /*!
         * \brief Execute the operator by starting all kernels on device.
         *
         * 1. wait on input as indicated by get_input_waiting_spec
         * 2. allocate memory for dynamic outputs
         * 3. call do_execute
         * 4. set_ready on output
         */
        void execute(ExecEnv &env) override final;

        /*!
         * \brief specifies waiting strategy on one comp node for input vars
         */
        struct InputWaitingSpecElem {
            //! on which comp node to wait other inputs
            CompNode comp_node;

            //! vars that must be ready on device
            VarNodeArray dev_ready;
        };

        using InputWaitingSpec = SmallVector<InputWaitingSpecElem, 1>;

        /*!
         * \brief get computing nodes that need to wait on other vars
         *
         * This is only valid after the computing func has been compiled.
         */
        const InputWaitingSpec& input_waiting_spec() const {
            return m_input_waiting_spec.val();
        }

        /*!
         * \brief set input waiting spec
         *
         * This should only be called from
         * SeqCompNodeOptimizerImpl::init_ready_event() or EagerEvalManager
         */
        void input_waiting_spec(InputWaitingSpec &&spec) {
            m_input_waiting_spec = std::move(spec);
        }

        /* =============== memory optimization =============== */

        /*!
         * \brief add layout constraint for input vars by calling
         *      VarNode::add_layout_constraint
         *
         * Note that this method is always called exactly once for operators
         * that are inserted into the computing sequence
         */
        virtual void add_input_layout_constraint() {}

        /*!
         * \brief called by graph compiler to setup readonly memory forwarding
         *
         * This function would always be called unless input has dynamic storage
         * but output has static storage
         */
        virtual void mem_plan_fwd_in2out_readonly() {}

        /*!
         * \brief called by graph compiler to setup writable memory forwarding
         *
         * This function would always be called unless input has dynamic storage
         * but output has static storage
         */
        virtual void mem_plan_fwd_in2out_writable() {}

        /* ===================== event callbacks ===================== */
        struct OprEventCallback;

        /*!
         * \brief get callbacks to be invoked on events related to this
         *      operator; default implementation returns empty event
         */
        virtual OprEventCallback get_opr_event_callback();

        /*!
         * \brief called when stream of comp node of output vars is changed for
         *      graph optimization
         */
        virtual void on_output_comp_node_stream_changed() = 0;

        /* ===================== initialization ===================== */

        /*!
         * \brief initialize output dtype by calling VarNode::dtype
         *
         * The default implementation requires all inputs to have the same dtype
         * and set output dtype to it
         *
         * This function is called once during operator insertion.
         */
        virtual void init_output_dtype();

        /*!
         * \brief initialize output format by calling VarNode::format
         *
         * The default implementation require all inputs to have the same
         * non-default format and set all non-volatile outputs format to it.
         *
         * This function is called once during operator insertion
         */
        virtual void init_output_format();

        /*!
         * \brief inititialize output comp_node by calling VarNode::comp_node
         *
         * This function is called once during operator insertion.
         */
        virtual void init_output_comp_node() = 0;

        /*!
         * \brief call VarNode::add_rt_force_dynamic_mem_alloc_imply_chain on
         *      input and output vars
         *
         * This function is called once during operator insertion.
         */
        virtual void init_rt_force_dynamic_mem_alloc_imply_chain() {}

        /*!
         * \brief register static infer descriptors for output vars by calling
         *      methods on ComputingGraph::static_infer_manager()
         *
         * This function is called once during operator insertion.
         */
        virtual void init_output_static_infer_desc() = 0;

        /*!
         * \brief initialize mem alloc plan for output nodes
         *
         * Mem plans are used for memory optimization; the storage of var node's
         * device tensor should always come from mem plan
         *
         * Default implmentation works by calling VarNode::init_mem_plan on vars
         * that match *dynamic* param
         *
         * output(...)->shape() is guaranteed to be valid before calling this
         * function.
         *
         * Remember to add Flag::IMPURE_OUTPUT_MEM_PLAN if needed.
         *
         * \param dynamic if true, initialize mem plans for vars that could not
         *      be statically inferred; otherwise for statically inferable vars
         */
        virtual void init_output_mem_plan(bool dynamic);

        /*
         * =============================================================
         * methods that should only be used by subclass or mixin classes
         * =============================================================
         */

        //! used by add_input() to sort vars for deduplication
        enum class AddInputSortType {
            NONE,
            CUR_ADDED,  //!< sort newly added vars
            ALL         //!< sort all currently added vars
        };

        //! add input var to this operator
        void add_input(std::initializer_list<VarNode*> list,
                AddInputSortType sort_type = AddInputSortType::NONE);

        /*!
         * \brief allocate a new output VarNode; the name would be appended to
         *      this->name to form the final name
         */
        VarNode* add_output(const Maybe<std::string> &name);

        /*!
         * \brief add extra component for equivalence check
         *
         * This is only a helper function to make the default hash() and
         * is_same() implementation consider other components in addition to all
         * the input nodes; you can also override hash() and is_same() to
         * implement deduplication.
         *
         * Note that the order for calling add_equivalence_component matters.
         * Also note that all input vars are used for deduplication by default.
         */
        template<typename T, typename ...Args>
        void add_equivalence_component(Args &&...args) {
            do_add_equivalence_component(
                    HashableContainer::create<T>(std::forward<Args>(args)...));
        }

        /*!
         * \brief allocate a new node prop and initialize dep entry as all
         *      inputs
         */
        virtual NodeProp* do_make_node_prop() const;

        /*!
         * \brief Update operator priority.
         *
         * This method would be invoked if and only if initializing
         * `m_node_prop` or after the graph optimizer modified the opr's
         * priority.
         * \return whether the priority would be changed.
         */
        virtual bool update_priority() const;

    protected:

        /*!
         * \param input_var_naming used for generating default node name
         */
        OperatorNodeBase(ComputingGraph *owner,
                const OperatorNodeConfig &config,
                const std::string &default_name,
                const VarNodeArrayView &input_var_naming);

        OperatorNodeBase(const CtorParamPack &param):
            OperatorNodeBase(param.owner, param.config, param.default_name,
                    param.input_var_naming)
        {}

        /*!
         * actually execute; all input and output have been checked, and the
         * subclasses only need to perform the actual computing
         */
        virtual void do_execute(ExecEnv &env) = 0;

    private:
        std::string m_name;

        //! user supplied config
        const OperatorNodeConfig m_config;

        bool m_inserted_in_graph = false;
        //! input vars
        VarNodeArray m_input;
        //! output vars; note that they are owned by this opr and freed in the
        //! destructor
        VarNodeArray m_output;
        SmallVector<HashableContainer> m_extra_equiv_comp;
        mutable Maybe<NodeProp> m_node_prop;
        Maybe<InputWaitingSpec> m_input_waiting_spec;

        void do_add_equivalence_component(HashableContainer &&hashable);

        bool is_same_st(const Hashable &rhs) const override final;
};

/*!
 * \brief struct to specify the callback function pointers for various operator
 *      events.
 *
 * This exists mainly for optimization: if a callback is not needed, related
 * surronding code would not be inserted into execution queue.
 */
struct OperatorNodeBase::OprEventCallback {
    using cbptr_t = thin_function<void()>;

    /*!
     * \brief called when memory status changed
     *
     * Memory status is defined by all layouts and addresses of DEV_VALUE deps
     * and outputs; if any of it changes, this callback would be called before
     * execution
     */
    Maybe<cbptr_t> on_mem_status_changed;
};

//! helper base class for operator mixins
class OperatorNodeMixinBase: public NonCopyableObj {
};

/*!
 * \brief mixin classes for operators
 *
 * each mixin class should come with an implementation class in mixin namespace
 * and a helper template glue class.
 *
 * The mixin implementation can be stateful and define new interface; the glue
 * class implement corresponding virtual function in OperatorNodeBase by thoses
 * provided by mixins
 */
namespace mixin {

//! check that base is OperatorNodeBase
template<class Base_>
class CheckBase {
    static_assert(std::is_base_of<OperatorNodeBase, Base_>::value,
            "Base must be OperatorNodeBase");
    public:
        using Base = Base_;
};

/*!
 * \brief used as MixinImpl template parameter for mixin glue classes when Impl
 *      class has been included in Base
 */
class EmptyMixinImpl {
};

/*!
 * \brief mixin for opeators that work on a single computing node
 */
class SingleCNOperatorNode: public OperatorNodeMixinBase {
    CompNode m_comp_node;

    protected:
        using NodeProp = OperatorNodeBase::NodeProp;
        using ExecEnv = OperatorNodeBase::ExecEnv;

        /*!
         * \brief infer output comp node and update the comp node of all ouput
         *      vars
         *
         * Note: the comp node stored in this mixin class is updated via
         * mixin_on_output_comp_node_stream_changed(), which is called from
         * opr.on_output_comp_node_stream_changed() invoked by this function.
         */
        static void mixin_init_output_comp_node(OperatorNodeBase &opr);

        /*!
         * \brief only infer output comp node, without modifying anything
         *
         * This implementation uses the comp node from input, requiring that at
         * least one input exists and they are all placed on the same comp node.
         * It also checks the comp node set in config.
         */
        static CompNode mixin_infer_output_comp_node(
                const OperatorNodeBase& opr, bool cross_mem);

        CompNode mixin_comp_node() const {
            return m_comp_node;
        }

        /*!
         * \brief initialize NodeProp with SINGLE_COMP_NODE, and setup
         *      dependency on input
         */
        NodeProp* mixin_do_make_node_prop(const OperatorNodeBase &opr) const;

        void mixin_do_execute(
                OperatorNodeBase &opr, OperatorNodeBase::ExecEnv &env);

        void mixin_on_output_comp_node_stream_changed(OperatorNodeBase &opr);

        /*!
         * \brief set comp node during initializing
         */
        void mixin_comp_node(OperatorNodeBase &opr, CompNode node);

        /*!
         * \brief override by subclass to perform raw computing; this function
         *      is already dispatched on corresponding stream in ExecEnv
         */
        virtual void scn_do_execute() = 0;

        ~SingleCNOperatorNode() = default;
};

/*!
 * \brief mixin class for implementing operators whose output shapes are
 *      completely determined by input shapes
 */
class OutshapePureByInshapeOpr: public OperatorNodeMixinBase {
    size_t m_nr_managed_outputs = 0;
    size_t m_inp_run_id = -1;
    TensorShapeArray m_out_shp;

    bool infer_desc(size_t out_idx,
            TensorShape &dest, const StaticInferInpVal &inp);

    protected:
        /*!
         * By default, all output vars would be managed by
         * OutshapePureByInshapeOprBase; call this function to set the number
         * of output vars that should be managed by this helper (they would be
         * the first vars of all output vars).
         */
        void mixin_set_nr_managed_outputs(OperatorNodeBase &opr, size_t nr);

        void mixin_init_output_static_infer_desc(OperatorNodeBase &opr);

        /*!
         * \brief get output shapes from input shapes
         * \param inp_shape current input shape; each element matches an input
         *      var
         * \param out_shape output shape; storage already allocated, and each
         *      element matches an output var
         */
        virtual void get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const = 0;

        ~OutshapePureByInshapeOpr();
};

/*!
 * \brief mixin class for operator whose all inputs and outputs are the same
 *      shape
 */
class IOSameShapeOperatorNode: public OutshapePureByInshapeOpr {
    protected:
        void get_output_var_shape(
                const TensorShapeArray &inp_shape,
                TensorShapeArray &out_shape) const override final;

        ~IOSameShapeOperatorNode() = default;
};

} // namespace mixin

//! glue class to apply mixin::SingleCNOperatorNode
template<class Base = OperatorNodeBase,
         class MixinImpl = mixin::SingleCNOperatorNode>
MGB_DEFINE_CLS_WITH_SUPER_TPL(
        SingleCNOperatorNode, mixin::CheckBase<Base>::Base,
        public MixinImpl) // {
    public:
        using NodeProp = typename Base::NodeProp;
        using ExecEnv = typename Base::ExecEnv;

        CompNode comp_node() const{
            return this->mixin_comp_node();
        }

        void comp_node(CompNode node) {
            this->mixin_comp_node(*this, node);
        }

    protected:
        using Base::Base;

        void init_output_comp_node() override {
            MixinImpl::mixin_init_output_comp_node(*this);
        }

        NodeProp* do_make_node_prop() const override {
            return this->mixin_do_make_node_prop(*this);
        }

        void do_execute(ExecEnv &env) override final {
            this->mixin_do_execute(*this, env);
        }

        //! note: subclasses overriding this function must call Super
        void on_output_comp_node_stream_changed() override {
            this->mixin_on_output_comp_node_stream_changed(*this);
        }
};

//! glue class to apply mixin::OutshapePureByInshapeOpr
template<class Base = OperatorNodeBase,
         class MixinImpl = mixin::OutshapePureByInshapeOpr>
class OutshapePureByInshapeOpr: public mixin::CheckBase<Base>::Base,
                                public MixinImpl {
    protected:
        using Base::Base;

        void set_nr_managed_outputs(size_t nr) {
            this->mixin_set_nr_managed_outputs(*this, nr);
        }

        void init_output_static_infer_desc() override {
            this->mixin_init_output_static_infer_desc(*this);
        }
};

template<class Impl = mixin::SingleCNOperatorNode>
using SingleCNOperatorNodeBaseT = SingleCNOperatorNode<OperatorNodeBase, Impl>;
using SingleCNOperatorNodeBase = SingleCNOperatorNodeBaseT<>;
using SingleCNOutshapePureByInshapeOprBase =
    OutshapePureByInshapeOpr<SingleCNOperatorNodeBase>;
using SingleCNIOSameShapeOperatorNodeBase = OutshapePureByInshapeOpr<
    SingleCNOperatorNodeBase, mixin::IOSameShapeOperatorNode>;
using OprNodeArray = SmallVector<OperatorNodeBase*>;

/*!
 * \brief define a final operator class
 *
 * Note that opening brace is included
 */
#define MGB_DEFINE_OPR_CLASS(_name, _base, ...) \
MGB_DEFINE_CLS_WITH_SUPER(_name final, _base ,##__VA_ARGS__) \
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

} // namespace cg
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
