/**
 * \file src/jit/include/megbrain/jit/compiler.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "internal_graph.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/utils/big_key_hashmap.h"

#if MGB_JIT

namespace mgb {
namespace jit {

/*!
 * \brief abstract Executable interface
 */
class Executable {
public:
    virtual ~Executable() noexcept = default;

    /*!
     * \brief execute the computation of a JIT opr
     *
     * Note that the same executable may be used for multiple JIT oprs. This
     * method may be called from multiple threads and it must be thread safe.
     */
    virtual void execute(JITExecutor* fusion_opr) = 0;
};

/*!
 * \brief abstract Compiler interface
 *
 * This is used by JITExecutor opr to create an Executable instance.
 */
class Compiler {
public:
    virtual ~Compiler() noexcept = default;

    //! static compiler properties
    struct Property {
        enum class Flag : uint32_t {
            NONE = 0,

            //! whether calling Elemwise::broadcast_collective_collapse is
            //! needed for the Args
            NEED_INPUT_COLLAPSE = 1u << 0,

            //! whether Executable needs identical ndim to be shared
            BIND_NDIM = 1u << 1,

            //! whether Executable needs identical shapes to be shared
            BIND_SHAPE = 1u << 2,

            //! if true, input would be contiguous; otherwise it is only
            //! monotone contiguous
            NEED_INPUT_CONTIG = 1u << 3
        };

        //! flags that indicate requirements of this Compiler for the
        //! JITExecutor opr
        Flag flag;

        //! supported features by this compiler
        JITFeatureBits feature_bits;

        //! maximal number of inputs for a fused opr
        size_t max_nr_input;

        inline bool contain_flag(Flag f) const;
    };

    static bool is_supported_device(CompNode::DeviceType device);

    /*!
     * \brief factory method to get an instance for a given device type
     *
     * The Compiler instances are associated with the graph. This method is
     * thread-safe.
     */
    static Compiler* get(ComputingGraph& graph, CompNode comp_node);

    /*!
     * \brief compile for a given operator
     *
     * This method is thread-safe.
     *
     * \return The compiled Executable, whose lifetime is managed by this
     *      Compiler instance.
     */
    Executable* compile(JITExecutor* opr);

    virtual Property property() const = 0;

    //! get number of execution workspace vars needed; called from ctor
    virtual size_t get_nr_workspace_outputs(JITExecutor* opr) const = 0;

    //! initialize satic infer for shapes of workspace outputs
    virtual void init_workspace_size_infer(JITExecutor* opr) = 0;

protected:
    /*!
     * \brief implemented by subclasses to do the compile when cache is
     *      unavailable.
     *
     * This call is protected in a mutex. Note that the returned Executable may
     * be used on multiple oprs with the same internal graph and args.
     */
    virtual std::unique_ptr<Executable> do_compile(
            const InternalGraph& graph, const JITExecutor::Args& args) = 0;

private:
    using ArgsCache = big_key_hash_map::BigKeyHashMap<
            std::unique_ptr<Executable>, JITExecutor::Args::HashEq,
            big_key_hash_map::Ref<JITExecutor::Args> >;

    using ExprCache =
            std::unordered_map<const InternalGraph*, ArgsCache,
                               InternalGraph::PtrHash, InternalGraph::PtrEqual>;

    class EmptyCompiler;

    ExprCache m_expr_cache;
    std::mutex m_mtx;
};

MGB_DEF_ENUM_CLASS_BIT_OPR(Compiler::Property::Flag);

bool Compiler::Property::contain_flag(Flag f) const {
    return static_cast<bool>(flag & f);
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
