/**
 * \file src/core/include/megbrain/graph/static_infer.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/tensor.h"

namespace mgb {

namespace imperative {
    class ProxyGraph;
} // namespace imperative

namespace cg {

class VarNode;

namespace static_infer {
    class StaticInferManagerImpl;

    //! Tag identifies the object on which to associate a value or shape; the
    //! actual content of the underlying VarNode is irrelevant here
    using Tag = VarNode*;

    /*!
     * \brief dependency type
     */
    enum class DepType {
        SHAPE,
        VALUE
    };

    /*!
     * \brief describes a single dependency
     */
    struct DepElement {
        Tag dest;
        DepType type;
    };
    using DepVal = SmallVector<DepElement>;

    /*!
     * \brief the actual value of a DepElement to be passed to infer funcs
     */
    class InpElement {
        const TensorShape *m_shape = nullptr;
        const DeviceTensorND *m_value = nullptr;

        friend class StaticInferManagerImpl;
        friend class imperative::ProxyGraph;

        public:
            /*!
             * \brief get the shape; this is always available regardless of
             *      what dep_type is
             */
            const TensorShape& shape() const {
                if (m_shape)
                    return *m_shape;
                return m_value->shape();
            }

            /*!
             * \brief get the inferred value; if dep_type is DepType::SHAPE, an
             *      exception would be thrown
             *
             * The tensor is placed on CompNode::default_cpu().
             * Note that value may be not contiguous.
             */
            const DeviceTensorND& value() const;
    };

    /*!
     * \brief input for a descriptor function
     */
    struct InpVal {
        //! same run_id implies identical val; starting from 1
        size_t run_id = 0;
        SmallVector<InpElement> val;
    };

    /*!
     * \brief type of an infer desc without deps
     */
    enum class SourceType {
        DEP,        //!< depends on others
        CONSTANT,   //!< no other deps; value would not change
        MUTABLE     //!< no other deps; value could be inferred before
                    //!  do_execute()
    };

    /*!
     * \brief descriptor for shape inference
     */
    struct ShapeInferDesc {
        SourceType src_type;
        DepVal deps;

        /*!
         * \brief func to do inference; return false if the shape could not
         *      be inferred now (only allowed so during graph initialization
         *      with src_type == MUTABLE)
         */
        using infer_func_t = thin_function<bool(
                TensorShape &dest, const InpVal &val)>;
        infer_func_t infer_func;

        /*!
         * \brief make a ShapeInferDesc that copies shape of another var into
         *      dest var
         */
        static ShapeInferDesc make_identity(VarNode *src);

        /*!
         * \brief make a constant ShapeInferDesc that always produces given
         *      value
         */
        static ShapeInferDesc make_const(const TensorShape &shp);
    };

    /*!
     * \brief descriptor for value inference
     */
    struct ValueInferDesc {
        SourceType src_type;
        DepVal deps;

        /*!
         * \brief func to do inference
         *
         * return false if the value could not be inferred now (only allowed
         * during graph initialization with src_type == MUTABLE). dest is
         * already placed on CompNode::default_cpu(), and dtype set to that of
         * corresponding var; its comp node should not be changed.
         *
         * Note: it is allowed to assign an input value directly to \p dest. In
         * such case, care must be taken if another execution of infer_func
         * might modify \p dest rather than assign it from some tensor.  Because
         * \p dest is cached, such modification would actually modify the input
         * value assigned to it.
         */
        using infer_func_t = thin_function<bool(
                DeviceTensorND &dest, const InpVal &val)>;
        infer_func_t infer_func;

        /*!
         * \brief make a ValueInferDesc that copies shape of another var into
         *      dest var
         */
        static ValueInferDesc make_identity(VarNode *src);
    };

    struct InferType {
        /*!
         * note: the enum values are defined in a bitwise manner to help
         * checking for existence (e.g. one can write flag & (CONST |
         * RT_STATIC)); the stored value can take one and only one flag
         */
        enum Flag: uint8_t {
            NO_DESC = 1 << 0,   //!< no infer desc has been registered
            CONST = 1 << 1,     //!< constant
            RT_STATIC = 1 << 2, //!< inferable before graph execution

            //! infer desc registered but some inputs are missing
            MISSING_INP = 1 << 3
        };

        //! infer type for shape and value; one of the above flags
        Flag shape, value;
    };

    /*!
     * \brief manager for statically inferring of var shapes and value on CPU
     *
     * Operators should register inference descriptors for their output vars.
     * Each inference descriptor can provide either shape or value. If it
     * depends on other vars, it must be pure (i.e. fully determinable from its
     * inputs)
     *
     * Infer desc on a var could not be registered if another desc depending on
     * it is registered. Inferred shapes would be used by memory allocating
     * sub-system, unless NO_SYS_MEM_ALLOC is specified.
     *
     */
    class StaticInferManager: public NonCopyableObj {
        public:

            virtual ~StaticInferManager() = default;

            /*!
             * \brief register an inference descriptor for shape of *dest*
             */
            virtual void register_shape_infer(
                    Tag dest, const ShapeInferDesc &desc) = 0;

            /*!
             * \brief register an inference descriptor for value of *dest*;
             *      shape infer must have been registered on this var
             *
             * Note that shape desc must be registered before value dep
             */
            virtual void register_value_infer(
                    Tag dest, const ValueInferDesc &desc) = 0;

            /*!
             * \brief get the type of static infer that could be performed on a
             *      var
             */
            virtual InferType get_infer_type(Tag dest) = 0;

            /*!
             * \brief get inferred shape of a var; if called before graph
             *      execution, the InferType must not be NO_DESC or MISSING_INP
             */
            virtual const TensorShape& infer_shape(Tag dest) = 0;

            /*!
             * \brief like infer_shape(), but allow MUTABLE source to return
             *      fail
             *
             * This method can be called before graph execution to try to get
             * the inferred shape.
             */
            virtual const TensorShape* infer_shape_fallible(Tag dest) = 0;

            /*!
             * \brief get inferred value of a var; if called before graph
             *      execution, the InferType must not be NO_DESC or MISSING_INP
             */
            virtual const DeviceTensorND& infer_value(Tag dest) = 0;

            /*!
             * \brief like infer_value(), but allow MUTABLE source to return
             *      fail
             *
             * This method can be called before graph execution to try to get
             * the inferred value.
             */
            virtual const DeviceTensorND* infer_value_fallible(Tag dest) = 0;

            /*!
             * \brief get source tags with RT_STATIC infer type needed to infer
             *      a tag; dest tag must be statically inferable
             */
            virtual DepVal get_rt_static_source_deps(
                    const DepElement &dest) = 0;
    };

    /*!
     * \brief a class to help update static inference manually
     *
     * Note that StaticInferManager::infer_shape and
     * StaticInferManager::infer_value would not check for the changes of
     * MUTABLE source nodes. To run static inference manually after source
     * changes, you need to use this StaticInferUpdater.
     */
    class StaticInferUpdater : public NonCopyableObj {
    public:
        static std::unique_ptr<StaticInferUpdater> make();

        virtual ~StaticInferUpdater() = default;

        //! add a target var of interest; it must be RT_STATIC type
        virtual StaticInferUpdater& add_dest(const DepElement& dest) = 0;

        /*!
         * \brief update internal status so static infer can reflect current
         *      value of mutable sources
         *
         * Note that var shapes would not be updated. Latest var shapes can only
         * be accessed via StaticInferManager::infer_shape().
         */
        virtual void update() = 0;
    };

    /*!
     * \brief static inference in subgraph, forwarding vars from par graph as
     *      input, and forward output vars into par graph
     *
     * Used for operators that maintain a subgraph; currently only used for
     * Loop.
     */
    class SubgraphStaticInferHelper: public NonCopyableObj {
        public:
            static std::unique_ptr<SubgraphStaticInferHelper> make();

            virtual ~SubgraphStaticInferHelper() = default;

            /*!
             * \brief register shape infer for an input var in sub graph
             *
             * Note that the caller operator needs to add deps in desc into its
             * OperatorNodeBase::dep_map() if needed.
             *
             * \param dest infer dest var; must be in the sub graph
             * \param desc shape infer desc; all deps must be in par graph
             */
            virtual void register_shape_infer_sub(Tag dest,
                    const ShapeInferDesc &desc) = 0;

            /*!
             * \brief register value infer for an input var in sub graph
             *
             * See register_shape_infer_sub for more details
             */
            virtual void register_value_infer_sub(Tag dest,
                    const ValueInferDesc &desc) = 0;

            /*!
             * \brief register shape infer for an output var in par graph
             * \param desc shape infer desc; deps could either be in par graph
             *      or sub graph
             * \return whether shape infer could be registered; it would be
             *      false if any dep is not statically inferable in par graph
             */
            MGB_WARN_UNUSED_RESULT
            virtual bool register_shape_infer_par(Tag dest,
                    const ShapeInferDesc &desc) = 0;

            /*!
             * \brief register value infer for an output var in par graph
             *
             * See register_shape_infer_par for more details
             */
            virtual bool register_value_infer_par(Tag dest,
                    const ValueInferDesc &desc) = 0;
    };

} // static_infer

using StaticInferInpVal = static_infer::InpVal;

} // cg
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

