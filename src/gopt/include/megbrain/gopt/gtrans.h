/**
 * \file src/gopt/include/megbrain/gopt/gtrans.h
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
#include "megbrain/utils/arith_helper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith.h"

namespace mgb {
namespace gopt {
    using cg::OperatorNodeBase;

    //! policy for defining whether a var is constant
    enum class ConstVarType {
        IMMUTABLE, IMMUTABLE_AND_PARAM
    };


    //! whether a var is given const value
    static inline bool is_const_value(SymbolVar var, float val) {
        auto v = var.as_immutable_scalar();
        return v.valid() && almost_equal(v->get_cast<float>(), val);
    }

    //! return non-null if owner_opr of var is type Op
    template<class Op>
    inline Op* try_cast_as_op(OperatorNodeBase *opr) {
        if (opr->same_type<Op>())
            return &opr->cast_final<Op>();
        return nullptr;
    }
    template<class Op>
    inline Op* try_cast_as_op(VarNode *var) {
        return try_cast_as_op<Op>(var->owner_opr());
    }

    //! if opr is Elemwise with given mode, return ii; otherwise return nullptr
    static inline opr::Elemwise* as_elem_opr(
            OperatorNodeBase *opr, opr::Elemwise::Mode mode) {
        if (auto op = try_cast_as_op<opr::Elemwise>(opr)) {
            if (op->param().mode == mode) {
                return op;
            }
        }
        return nullptr;
    }
    static inline opr::Elemwise* as_elem_opr(
            VarNode *var, opr::Elemwise::Mode mode) {
        return as_elem_opr(var->owner_opr(), mode);
    }

    //! whether an opr is SharedDeviceTensor or ImmutableTensor
    static inline bool is_const_var(
            ConstVarType policy, OperatorNodeBase *opr) {
        auto type = opr->dyn_typeinfo();
        return type == opr::ImmutableTensor::typeinfo() ||
               (policy == ConstVarType::IMMUTABLE_AND_PARAM &&
                (type == opr::SharedDeviceTensor::typeinfo() ||
                 type == opr::MultipleDeviceTensorHolder::typeinfo()));
    }

    //! whether an operator is binary opr and its inputs are commutable
    static inline bool is_commutable_binary(OperatorNodeBase *opr) {
        return opr->same_type<opr::Elemwise>() &&
            ::megdnn::Elemwise::ModeTrait::from_mode(
                    opr->cast_final<opr::Elemwise>().param().mode).commutable;
    }

    //! return the var node if opr has single output var, and nullptr otherwise
    VarNode* get_opr_single_output_var(OperatorNodeBase *opr);

    //! result of graph transformation
    struct GTransResultItem {
        //! transformation result
        VarNode *result = nullptr;
        //! human readable message describing the rule applied
        const char *msg = nullptr;
        //! internal vars that result depend on, which might be further
        //! transformed; may be null
        std::array<VarNode*, 2> internal{{nullptr, nullptr}};

        GTransResultItem() = default;
        GTransResultItem(
                VarNode *r, const char *m,
                std::initializer_list<VarNode*> i):
            result{r}, msg{m}
        {
            mgb_assert(i.size() <= internal.size());
            std::copy(i.begin(), i.end(), internal.begin());
        }
    };
    using GTransResult = Maybe<GTransResultItem>;

    /*!
     * \brief visit a subtree defined by \p check_internal
     * \param check_internal predicate function that check whether owner opr of
     *      the var should be considered as an internal node, so it would be
     *      further expanded; must return false for zero-input or
     *      multiple-output oprs, so the tree can be well defined; it is called
     *      exactly once for every node in the tree (note that the nodes may
     *      contain duplicated opr/var pointers)
     * \param on_leaf called when a leaf node is encountered; can be null
     * \param on_internal_finish callback when all children of an internal node
     *      has been visited
     * \param allow_multi_cn whether to allow oprs that have inputs on
     *  `   different comp nodes to be considered as internal node
     */
    void visit_opr_tree(
            VarNode *endpoint,
            const thin_function<bool(VarNode*)> &check_internal,
            const thin_function<void(VarNode*)> &on_leaf = {},
            const thin_function<void(OperatorNodeBase*)> &
                on_internal_finish = {},
            bool allow_multi_cn = false);

    /*!
     * \brief extract list of leaf nodes that do not satisfy predicate
     * \param pred callable to check whether a var should be considered as
     *      internal node
     * \param allow_multi_cn whether to allow oprs that have inputs on
     *  `   different comp nodes to be considered as internal node
     * \return list of leaf vars in DFS order; note that there may be
     *      duplications
     */
    VarNodeArray extract_opr_leaves(
            VarNode *endpoint,
            const std::function<bool(OperatorNodeBase*)> &pred,
            bool allow_multi_cn = false);

    /*!
     * \brief reduce var list by the applying elemwise opr with given mode
     * \param mode reduce mode; must be binary opr
     * \param mid_results if not null, the intermediate results would be
     *      appended to this array
     */
    VarNode* elemwise_reduce_var_list(
            const VarNodeArray &vars,
            opr::Elemwise::Mode mode,
            VarNodeArray *mid_results = nullptr);


    /*!
     * \brief algebra transformation applied on sub-expression f(g(a, b), c)
     *
     * 2 means unpack 2 binary oprs, and 0 means second unpacked opr is the 0th
     * input of first opr.
     */
    class BinaryTrans20: NonCopyableObj {
        class Rule;
        class AssociativeRuleReg;
        class DistributiveAddRuleReg;

        std::unordered_map<std::pair<Typeinfo*, Typeinfo*>, Rule*, pairhash>
            m_rules;

        BinaryTrans20() = default;

        public:

            //! f(g(a, b), c) => f1(a, g1(b, c))
            static BinaryTrans20& associtive();

            //! f(a + b, c) => f1(a, c) + f2(b, c)
            static BinaryTrans20& distributive_add();

            /*!
             * \brief try to apply the transform given *f* and *g* operators
             * \param fop *f* given in the definition
             * \param swap_fop_inp if true, then fop must be commutable gop
             *      must be its second input
             * \param swap_gop_inp if true, then gop must be commutable and
             *      its inputs would be swapped to consider vars a and b
             */
            GTransResult apply(OperatorNodeBase *fop,
                    bool swap_fop_inp = false, bool swap_gop_inp = false) const;
    };

    /*!
     * \brief check whether x == inv(y) for group specified by mode
     * \tparam mode ADD or MUL, where inv corresponds to NEGATE or POW(., -1)
     *      respectively
     * \return y if x == inv(y), or nullptr otherwise
     */
    template<opr::Elemwise::Mode mode>
    VarNode* check_is_group_inverse_opr(SymbolVar x);

    template<>
    inline VarNode* check_is_group_inverse_opr<opr::Elemwise::Mode::ADD>(
            SymbolVar x) {
        auto opr = as_elem_opr(
                x.node()->owner_opr(), opr::Elemwise::Mode::NEGATE);
        return opr ? opr->input(0) : nullptr;
    }

    //! helper for hash of TensorShape
    class TensorShapeHashKey {
        const TensorShape m_shape;
        const size_t m_hash = 0;

        public:
            TensorShapeHashKey() : m_shape() {}

            TensorShapeHashKey(const TensorShape &shp):
                m_shape{shp},
                m_hash{static_cast<size_t>(XXHash().
                        update(&shp.ndim, sizeof(shp.ndim)).
                        update(shp.shape, sizeof(shp.shape[0]) * shp.ndim).
                        digest())
                }
            {
            }

            const TensorShape& shape() const {
                return m_shape;
            }

            bool operator == (const TensorShapeHashKey &rhs) const {
                return m_hash == rhs.m_hash && m_shape.eq_shape(rhs.m_shape);
            }

            struct Hash {
                size_t operator() (const TensorShapeHashKey &key) const {
                    return key.m_hash;
                }
            };

            using Pair = std::pair<TensorShapeHashKey, TensorShapeHashKey>;

            struct PairHash {
                size_t operator() (const Pair &key) const {
                    return hash_pair_combine(
                            key.first.m_hash, key.second.m_hash);
                }
            };

            template<typename R>
            using Map = std::unordered_map<TensorShapeHashKey, R, Hash>;

            template<typename R>
            using PairMap = std::unordered_map<Pair, R, PairHash>;
    };
} // namespace gopt
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
