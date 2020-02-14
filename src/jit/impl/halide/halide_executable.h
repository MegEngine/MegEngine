/**
 * \file src/jit/impl/halide/halide_executable.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./halide_header.h"

#if MGB_JIT_HALIDE

#include "./ast_hl.h"
#include "megbrain/jit/compiler.h"

#include <atomic>

namespace mgb {
namespace jit {

/*!
 * \brief JIT executable with Halide backend
 *
 * This class handles the translation of InternalGraph to halide func and
 * calling Halide functions. A compiler for a specific platform should implement
 * a TargetTrait to create executables.
 */
class HalideExecutable final : public Executable {
public:
    //! a handle for calling Halide functions; this class is move-only
    struct FunctionHandle {
        //! map for user context
        struct UctxMap {
            //! Halide user context to represent mgb comp node.
            //! Its memory is managed by TargetTrait
            CompNode::UnorderedMap<void*> cn2uctx;
            std::mutex mtx;
        };

        //! TargetTrait impls should call init_uctx_map() if user context is
        //! needed
        UctxMap* uctx_map = nullptr;

        //! handle to be passed to dlclose()
        void* dl_handle = nullptr;

        halide_device_interface_t* (*get_device_interface)() = nullptr;

        //! execute the actual computing.
        //! The arguments are (user_context, inputs..., outputs...)
        int (*execute)(void** argv) = nullptr;

        //! callback to release the device in dtor; can be null
        int (*device_release)(void* user_context) = nullptr;

        void swap(FunctionHandle& rhs) {
            using T = std::aligned_storage_t<sizeof(FunctionHandle),
                                             alignof(FunctionHandle)>;
            T tmp;
            tmp = *reinterpret_cast<T*>(this);
            *reinterpret_cast<T*>(this) = reinterpret_cast<T&>(rhs);
            reinterpret_cast<T&>(rhs) = tmp;
        }

        //! called by TargetTrait impls if user context is needed
        void init_uctx_map() {
            mgb_assert(!uctx_map);
            uctx_map = new UctxMap;
        }

        FunctionHandle() = default;
        FunctionHandle(FunctionHandle&& rhs) { swap(rhs); }
        FunctionHandle(const FunctionHandle&) = delete;
        FunctionHandle& operator=(FunctionHandle&& rhs) {
            swap(rhs);
            return *this;
        }
        FunctionHandle& operator=(const FunctionHandle&) = delete;

        ~FunctionHandle();
    };

    /*!
     * \brief user data to be associated with a HalideExecutable
     *
     * This is needed since multiple HalideExecutable objects may share the
     * TargetTrait object.
     */
    struct TargetTraitUserData {
        virtual ~TargetTraitUserData() = default;
    };

    //! to be implemented by subclass for a specific device type
    class TargetTrait {
    public:
        using FunctionHandle = HalideExecutable::FunctionHandle;
        using FeatureSet = std::bitset<Halide::Target::FeatureEnd>;

        virtual ~TargetTrait() = default;

        /*!
         * \brief Halide features needed for this computing platform
         *
         * JITFusion oprs with the same features would share the underlying
         * FunctionHandle
         */
        virtual FeatureSet features(CompNode comp_node) const = 0;

        //! get user context for a comp node
        virtual void* get_user_context(CompNode comp_node) = 0;

        //! compile and load a Halide function; it must be thread safe
        virtual FunctionHandle compile_and_load(
                CompNode comp_node, Halide::Target halide_target,
                const HalideExecutable& hl_exec) = 0;

    protected:
        /*!
         * \brief get the user data associated with a HalideExecutable
         * \param maker the callback to be invoked if user data has not been
         *      created
         */
        TargetTraitUserData* user_data(
                const HalideExecutable& hl_exec,
                thin_function<std::unique_ptr<TargetTraitUserData>()> maker);
    };

    HalideExecutable(std::shared_ptr<TargetTrait> trait,
                     const InternalGraph& graph, const JITExecutor::Args& args);
    ~HalideExecutable();

    void execute(JITExecutor* fusion_opr) override;

    //! get the inputs for the Halide function
    std::vector<Halide::Argument> halide_inputs() const;

    //! get output var for the Halide function
    const ast_hl::AstNodePtr& halide_output() const { return m_halide_output; }

    static halide_type_t dtype_mgb2halide(DType dtype);

private:
    std::shared_ptr<TargetTrait> const m_target_trait;
    ast_hl::AstNodePtr m_halide_output;

    //! index of input var and corresponding halide InputDevValueOp
    SmallVector<std::pair<size_t, ast_hl::AstNodePtr>> m_value_inputs;

    std::mutex m_mtx;
    std::unordered_map<TargetTrait::FeatureSet,
                       std::pair<std::mutex, FunctionHandle>>
            m_feature_set2func;
    CompNode::UnorderedMap<std::atomic<FunctionHandle*>> m_cn2func;

    mutable std::unique_ptr<TargetTraitUserData> m_target_trait_user_data;
    mutable std::mutex m_target_trait_user_data_mtx;

    void invoke(void* user_context, const FunctionHandle& handle,
                const VarNodeArray& inputs, VarNode* output);
    static ast_hl::AstNodePtr mgb_var_to_halide_buffer(VarNode* var);

    //! prepare args and call TargetTrait::compile_and_load for given comp node
    FunctionHandle compile_and_load(CompNode comp_node) const;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_HALIDE

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
