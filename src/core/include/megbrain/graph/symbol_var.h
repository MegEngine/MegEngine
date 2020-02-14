/**
 * \file src/core/include/megbrain/graph/symbol_var.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/var_node.h"

namespace mgb {
namespace cg {

/*!
 * \brief Wrap a VarNode* for operator overloading
 */
class SymbolVar {
    VarNode *m_node = nullptr;

    public:
        SymbolVar() = default;

        SymbolVar(cg::VarNode *node): m_node(node)
        {}

        cg::VarNode* node() const {
            return m_node;
        }

        DType dtype() const {
            return m_node->dtype();
        }

        /*!
         * \brief if the value is immutable and equals to some value at every
         *      position, return it
         *
         * Note: the shape may be larger than (1, )
         */
        Maybe<DTypeScalar> as_immutable_scalar() const;

        //! similar to as_immutable_scalar(), but also require shape to be (1, )
        Maybe<DTypeScalar> as_immutable_scalar_require_shape() const;

        /*!
         * \brief insert an operation with signle output into underlying graph
         * Implemented in graph/cg.h
         */
        template<typename Node, typename ...Args>
        inline SymbolVar insert_single_output_opr(Args&& ...args) const;

        /*!
         * \brief set a new name; note that the underlying VarNode would be
         *      modified, not this SymbolVar itself
         */
        SymbolVar rename(const std::string &name) const;

        SymbolVar reshape(const TensorShape &tshape) const;
        SymbolVar reshape(SymbolVar tshape) const;
        SymbolVar broadcast(const TensorShape &tshape) const;
        SymbolVar broadcast(SymbolVar tshape) const;
        SymbolVar symshape() const;
        SymbolVar flatten() const;

        const TensorShape& shape() const {
            return m_node->shape();
        }

        TensorFormat format() const {
            return m_node->format();
        }

        SymbolVar operator + (const SymbolVar &rhs) const;
        SymbolVar operator - (const SymbolVar &rhs) const;
        SymbolVar operator * (const SymbolVar &rhs) const;
        SymbolVar operator / (const SymbolVar &rhs) const;
        SymbolVar operator < (const SymbolVar &rhs) const;
        SymbolVar operator <= (const SymbolVar &rhs) const;
        SymbolVar operator > (const SymbolVar &rhs) const {
            return rhs < *this;
        }
        SymbolVar operator - () const;

#define DEF_CT_OPR(_op) \
        template<typename ctype> \
        typename ctype_enable_if<ctype, SymbolVar>::type \
        operator _op (ctype v) const { \
            return *this _op make_scalar(v); \
        }
        DEF_CT_OPR(+)
        DEF_CT_OPR(-)
        DEF_CT_OPR(*)
        DEF_CT_OPR(/)
        DEF_CT_OPR(>)
        DEF_CT_OPR(<)
        DEF_CT_OPR(<=)
#undef DEF_CT_OPR

        /*!
         * \brief fill the tensor with a constant value, but retaining the dtype
         */
        template<typename ctype>
        typename ctype_enable_if<ctype, SymbolVar>::type
        fill_retain_dtype(ctype val) const {
            DTypeScalar dval{dtype()};
            dval.set_retain_dtype(val);
            return make_scalar(dval).broadcast(symshape());
        }

        /*!
         * \brief make a const scalar value on given computing graph and
         *      computing node
         */
        static SymbolVar make_scalar(
                DTypeScalar value, ComputingGraph &cg, CompNode cn);

        /*!
         * \brief make a const scalar value using computing graph and comp node
         *      provided by this var
         */
        SymbolVar make_scalar(DTypeScalar value) const {
            return make_scalar(
                    value, *node()->owner_graph(), node()->comp_node());
        }

        /*!
         * \brief make a scalar with given value and dtype of this symvar
         */
        template<typename ctype>
        typename ctype_enable_if<ctype, SymbolVar>::type
        make_scalar_dt(ctype val) const {
            DTypeScalar dval{dtype()};
            dval.set_retain_dtype(val);
            return make_scalar(dval);
        }

        /*!
         * \brief get value in eager evaluation mode
         *
         * This essentially synchronizes the dispatch queue and then call
         * dev_tensor()
         */
        const DeviceTensorND& eager_eval_get_value() const;

        bool allow_shape_change() const {
            return m_node->allow_shape_change();
        }
};

using SymbolVarArray = SmallVector<SymbolVar>;

class SymbolVarArrayView;

/*!
 * \brief View SymbolVarArray or VarNodeArray as VarNode* list.
 *
 * This class is intended for passing a list of VarNode* in function parameters,
 * so unnecessary copy/conversion between VarNodeArray and SymbolVarArray can be
 * avoided.
 */
class VarNodeArrayView final : NonCopyableObj {
    static_assert(sizeof(SymbolVar) == sizeof(VarNode*), "bad size");
    static_assert(alignof(SymbolVar) == alignof(VarNode*), "bad align");
    VarNode* const* m_begin = nullptr;
    VarNode* const* m_end = nullptr;

    void check_idx(size_t idx) const;

public:
    VarNodeArrayView() = default;

    VarNodeArrayView(const VarNodeArray& arr)
            : m_begin{arr.data()}, m_end{m_begin + arr.size()} {}

    VarNodeArrayView(const SymbolVarArray& arr)
            : m_begin{reinterpret_cast<VarNode* const*>(arr.data())},
              m_end{m_begin + arr.size()} {}

    VarNodeArrayView(VarNode* const* begin, VarNode* const* end)
            : m_begin{begin}, m_end{end} {}

    template <size_t nr>
    VarNodeArrayView(const std::array<SymbolVar, nr>& arr)
            : m_begin{reinterpret_cast<VarNode* const*>(arr.data())},
              m_end{m_begin + arr.size()} {}

    inline explicit VarNodeArrayView(const SymbolVarArrayView& arr);

    VarNodeArrayView(std::initializer_list<VarNode*> s)
            : m_begin{s.begin()}, m_end{s.end()} {}

    VarNodeArrayView(std::initializer_list<SymbolVar> s)
            : m_begin{reinterpret_cast<VarNode* const*>(s.begin())},
              m_end{m_begin + s.size()} {}

    VarNode* operator[](size_t idx) const { return m_begin[idx]; }

    VarNode* at(size_t idx) const {
        check_idx(idx);
        return m_begin[idx];
    }

    size_t size() const { return m_end - m_begin; }

    bool empty() const { return m_begin == m_end; }

    VarNode* const* begin() const { return m_begin; }

    VarNode* const* end() const { return m_end; }
};

/*!
 * \brief Similar to VarNodeArrayView, but accessors return SymbolVarArray
 *      instead.
 *
 * Note: Implicit conversion only works from VarNodeArrayView to
 * SymbolVarArrayView. This is because the preferred use of SymbolVarArrayView
 * is for easily accessing items as SymbolVar when the parameter type is
 * VarNodeArrayView.
 */
class SymbolVarArrayView final : NonCopyableObj {
    SymbolVar const* m_begin = nullptr;
    SymbolVar const* m_end = nullptr;

    void check_idx(size_t idx) const;

public:
    SymbolVarArrayView(const VarNodeArrayView& arr)
            : m_begin{reinterpret_cast<SymbolVar const*>(arr.begin())},
              m_end{m_begin + arr.size()} {}

    SymbolVarArrayView(std::initializer_list<SymbolVar> s)
            : m_begin{s.begin()}, m_end{s.end()} {}

    SymbolVar operator[](size_t idx) const { return m_begin[idx]; }

    SymbolVar at(size_t idx) const {
        check_idx(idx);
        return m_begin[idx];
    }

    size_t size() const { return m_end - m_begin; }

    bool empty() const { return m_begin == m_end; }

    SymbolVar const* begin() const { return m_begin; }

    SymbolVar const* end() const { return m_end; }
};

VarNodeArrayView::VarNodeArrayView(const SymbolVarArrayView& arr)
        : m_begin{reinterpret_cast<VarNode* const*>(arr.begin())},
          m_end{m_begin + arr.size()} {}

#define DEF_CT_OPR(_op) \
    template<typename ctype> \
    typename ctype_enable_if<ctype, SymbolVar>::type \
    operator _op (ctype lhs, const SymbolVar &rhs) { \
        return rhs.make_scalar(lhs) _op rhs; \
    }
    DEF_CT_OPR(+)
    DEF_CT_OPR(-)
    DEF_CT_OPR(*)
    DEF_CT_OPR(/)
    DEF_CT_OPR(>)
    DEF_CT_OPR(<)
    DEF_CT_OPR(<=)
#undef DEF_CT_OPR

} // namespace cg
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

