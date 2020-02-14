/**
 * \file src/core/include/megbrain/graph/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/cg.h"
#include <vector>

namespace mgb {
namespace cg {

class OperatorNodeBase;
class VarNode;

/*!
 * \brief get the involved comp nodes of an operator; the operator must have
 *      been compiled
 */
CompNode::UnorderedSet get_opr_comp_node_set(OperatorNodeBase *opr);

/*!
 * \brief whether var shape could be statically inferred
 */
static inline bool is_static_var_shape(VarNode *var) {
    using IT = static_infer::InferType;
    auto it = var->owner_graph()->static_infer_manager().
        get_infer_type(var);
    return it.shape & (IT::CONST | IT::RT_STATIC);
}

/*!
 * \brief whether var shape is constant
 */
static inline bool is_const_var_shape(VarNode *var) {
    using IT = static_infer::InferType;
    auto it = var->owner_graph()->static_infer_manager().
        get_infer_type(var);
    return it.shape & IT::CONST;
}

/*!
 * \brief whether var value could be statically inferred
 */
static inline bool is_static_var_value(VarNode *var) {
    using IT = static_infer::InferType;
    auto it = var->owner_graph()->static_infer_manager().
        get_infer_type(var);
    return it.value & (IT::CONST | IT::RT_STATIC);
}

/*!
 * \brief whether var value is constant
 */
static inline bool is_const_var_value(VarNode* var) {
    using IT = static_infer::InferType;
    auto&& mgr = var->owner_graph()->static_infer_manager();
    auto infer_type = mgr.get_infer_type(var);
    if (!(infer_type.value & IT::CONST))
        return false;

    mgb_assert(infer_type.shape & IT::CONST,
               "var(%s) has const value infer but non-const shape infer",
               var->cname());

    return true;
}

/*!
 * \brief whether var storage would be statically allocated by system
 */
static inline bool is_static_var_storage(VarNode *var) {
    using F = VarNode::Flag;
    if (var->contain_flag(F::PERSISTENT_DEVICE_VALUE))
        return true;
    if (var->contain_flag(
                F::RT_FORCE_DYNAMIC_MEM_ALLOC | F::NO_SYS_MEM_ALLOC |
                F::NO_SYS_STATIC_MEM_ALLOC))
        return false;
    return is_static_var_shape(var);
}

/*!
 * \brief whether device computing is needed for given input var and dep type of
 *      an operator
 *
 * See the code for precise definition
 */
static inline bool need_device_computing_on_var(
        VarNode *var, OperatorNodeBase::NodeProp::DepType dt) {
    using DT = OperatorNodeBase::NodeProp::DepType;
    return !var->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE) &&
        ((dt & (DT::DEV_VALUE | DT::DEV_COMP_ORDER)) ||
        ((dt & DT::HOST_VALUE) && !is_static_var_value(var)) ||
        ((dt & DT::SHAPE) && is_static_var_shape(var)));
}

/*!
 * \brief whether all input vars of an operator has static storage
 */
bool is_all_input_static_storage(OperatorNodeBase* opr);

/*!
 * \brief transform a SymbolVarArray to a VarNodeArray
 */
VarNodeArray to_var_node_array(const SymbolVarArray& symbol_var_array);

/*!
 * \brief transform a VarNodeArray to a SymbolVarArray
 */
SymbolVarArray to_symbol_var_array(const VarNodeArray& var_node_array);

/*!
 * \brief return a string to describe the list of variables
 */
std::string dump_var_info(const VarNodeArrayView &vars);

/*!
 * \brief compute grad of target w.r.t. wrt (i.e. d(target)/d(wrt))
 * \param warn_mid_wrt whether to give warning on wrt not being end-point var
 * \param return_zero_for_nodep if *target* does not depend on *wrt*, return a
 *      zero-valued var rather than a null var
 * \return the var representing grad, or nullptr if target does not depend on
 *      wrt
 */
SymbolVar grad(SymbolVar target, SymbolVar wrt,
        bool warn_mid_wrt = true, bool return_zero_for_nodep = true);

/*!
 * \brief equivalant to calling grad(grad, wrt) one by one if symbolic;
 * since cache in grad manager would be cleared each time, this method is more
 * efficient if eager.
 */
SymbolVarArray grad(SymbolVar target, SymbolVarArray wrts,
        bool warn_mid_wrt = true, bool return_zero_for_nodep = true);

/*!
 * \brief get current grad target, which must be called inside
 *      OperatorNodeBase::grad() implementations
 */
SymbolVar current_grad_target(ComputingGraph &graph);

struct SpecialOprStat {
    bool has_virtual_grad = false;
};

/*!
 * \brief replace variables in a graph
 * \param dest target vars to describe the graph
 * \param varmap map that describes how to replace an old var with a new var
 * \return a list of vars correpsonding to \p dest whose dependencies have been
 *         replaced according to \p varmap
 */
SymbolVarArray replace_vars(const SymbolVarArray &dest,
        const ThinHashMap<SymbolVar, SymbolVar>& varmap);

/*!
 * \brief replace operator in a graph
 * \param dest target vars to describe the graph
 * \param oprmap map that describes how to replace an old operator with a new
 *        operator
 * \return a list of vars correpsonding to \p dest whose dependencies have been
 *         replaced according to \p oprmap
 */
SymbolVarArray replace_oprs(
        const SymbolVarArray& dest,
        const ThinHashMap<OperatorNodeBase*, OperatorNodeBase*>& oprmap);

/*!
 * \brief replace computing graph which owns all variables to another graph
 * \param dest target vars to describe the graph
 * \param new_graph target computing graph
 * \return a list of vars correpsonding to \p dest whose owner_graph have been
 *         replaced with \p new_graph
 */
SymbolVarArray replace_vars_comp_graph(
    const SymbolVarArray &dest, ComputingGraph* new_graph);


SymbolVarArray find_h2d(const SymbolVarArray& dest);

/*!
 * \brief go through OperatorNodeBase::NodeProp::Attribute::src_opr until it
 *      becomes nullptr
 *
 * This function also performs path compression
 */
OperatorNodeBase* get_opr_root_source_opr(OperatorNodeBase *opr);

//! describes how two mem plans intersect
enum class MemPlanIntersectionType {
    DISJOINT,   //!< no intersection
    IDENTICAL,  //!< completely same
    OVERLAP     //!< intersects but not identical
};
MemPlanIntersectionType get_mem_plan_intersection_type(VarNode* a, VarNode *b);

/*!
 * \brief request output var to writable forward input var if no mem plan of
 *      other input vars intersects with this input var
 */
void request_fwd_in2out_writable_if_no_mem_ovelap(
        OperatorNodeBase *opr, size_t inp, size_t out);


/*!
 * \brief update shapes of output vars; set to empty if not statically
 *      inferable
 *
 * This method must always be called if a new operator is inserted (currently
 * used in ComputingGraph::insert_opr and copy_opr_shallow)
 *
 * Note: implemented in cg_impl.cpp, since it is used during graph init
 */
void update_output_var_shapes(OperatorNodeBase *opr);

/*!
 * \brief add an output to be used as the workspace for an operator
 *
 * The workspace var would have dtype Byte.
 *
 * This helper is usually called from an opr constructor and used for adding the
 * last output.
 */
void add_workspace_output(OperatorNodeBase *opr);

/*!
 * \brief copy a raw tensor shape into a host tensor
 */
void copy_shape_to_tensor_value(DeviceTensorND &dest, const TensorShape &shp);

/*!
 * \brief copy value of a host tensor into a raw tensor shape
 */
void copy_tensor_value_to_shape(TensorShape &dest, const DeviceTensorND &val);

/*!
 * \brief get a symbolvar whose value is tensor shape, used for other
 *      operators
 *
 * \param opr_name operator that invokes this function; used in error
 *      function if *config* is invalid
 */
SymbolVar var_from_tensor_shape(
        ComputingGraph &graph, const OperatorNodeConfig &config,
        const char *opr_name,
        const TensorShape &shape);

/*!
 * \brief get a symbolvar whose value is tensor shape
 *
 * \param inp used to determine the computing graph, which can be any symbolvar
 *      belonging to the same computing graph.
 */
static inline SymbolVar var_from_tensor_shape(
        SymbolVar inp, const TensorShape &shape) {
    return var_from_tensor_shape(*inp.node()->owner_graph(),
            OperatorNodeConfig().follow_comp_node(inp),
            nullptr, shape);
}

/*!
 * \brief iterate over all dependency oprs in topological order
 * \param cb callback to be invoked when a new operator is discovered
 */
class DepOprIter {
    public:
        using Callback = thin_function<void(OperatorNodeBase*)>;
        using ExtraDep = ThinHashMap<OperatorNodeBase*, SmallVector<VarNode*>>;

        explicit DepOprIter(Callback cb,
                            std::shared_ptr<ExtraDep> extra_dep = nullptr)
                : m_cb{std::move(cb)}, m_extra_dep(std::move(extra_dep)) {}

        //! add an operator whose deps should be discovered
        void add(OperatorNodeBase *dest);

        void add(SymbolVar var) { add(var.node()->owner_opr()); }

        //! graph of all the oprs
        ComputingGraph* owner_graph() const {
            return m_owner_graph;
        }

        //! check if an opr has been visited
        bool visited(OperatorNodeBase *opr) const {
            return m_visited.count(opr);
        }

        //! set an opr to have been visited
        DepOprIter& set_visited(OperatorNodeBase* opr) {
            m_visited.insert(opr);
            return *this;
        }

    private:
        //! a single stack frame to avoid recursion
        struct Frame {
            OperatorNodeBase *opr;
            VarNode * const *inputs;
            VarNode * const *extra_deps;
            size_t inp_idx, nr_input, nr_extra_dep;
        };
        ComputingGraph *m_owner_graph = nullptr;
        std::vector<Frame> m_stack;
        ThinHashSet<OperatorNodeBase*> m_visited;
        Callback m_cb;
        const std::shared_ptr<ExtraDep> m_extra_dep;

        inline void push_stack(OperatorNodeBase *opr);

};

/*!
 * \brief a user data associated with ComputingGraph::Options::user_data
 *
 * When a graph A is copied as a new graph B, the module that initiates the copy
 * may associate an instance of InterGraphVarTransformer with user data of B, so
 * when B is exetended (e.g. by constructing a grad graph), others can know how
 * to transform a var in A into its equivalent var in B.
 */
class InterGraphVarTransformer final: public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    InterGraphVarTransformer() = default;

    public:

        /*!
         * var transforming function to be defined by copier; the input var has
         * been checked to be in src graph.
         */
        using TransFunc = thin_function<VarNode*(VarNode*)>;

        /*!
         * \brief register a transfomer to *dest* graph that takes var in *src*
         *      and outputs a corresponding var in *dest*
         *
         * This function should be called only once on a graph
         */
        static void register_to(ComputingGraph *dest,
                const ComputingGraph *src, const TransFunc &trans);

        /*!
         * \brief get the transformer associated with a graph
         * \return previously registered transformer on given graph or nullptr
         *      if none registered
         */
        static const InterGraphVarTransformer* get(const ComputingGraph &graph);

        /*!
         * \brief transform a var into this graph
         */
        VarNode *trans(VarNode *src) const;

    private:
        ComputingGraph *m_graph_dest;
        const ComputingGraph *m_graph_src;
        TransFunc m_trans_func;
};

/*!
 * \brief find extra dependency of vars (ComputingGraph::Options::extra_vardeps)
 *      and merge into a var list
 */
class ExtraDependencyMerger {
    SpecialOprStat* const m_sopr_stat;
    VarNodeArray m_new_deps;
    DepOprIter m_opr_iter;
    SymbolVarArray m_result;
    ComputingGraph* m_owner_graph = nullptr;

    void on_opr(OperatorNodeBase* opr);

public:
    explicit ExtraDependencyMerger(SpecialOprStat* sopr_stat = nullptr);
    ~ExtraDependencyMerger();

    /*!
     * \brief add a new set of vars
     * \return current var list after adding this vars. It keeps growing.
     *
     * Note: \p vars given here would always be added to the result list, even
     * if they duplicate existing vars.
     *
     * \return vars with extra dependency; the returned list can be modified
     */
    SymbolVarArray& add(const SymbolVarArray& vars);
};

//! shortcut for calling ExtraDependencyMerger
SymbolVarArray get_dest_vars_with_extra_deps(
        const SymbolVarArray& dest_vars, SpecialOprStat* sopr_stat = nullptr);

} // cg
} //mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
