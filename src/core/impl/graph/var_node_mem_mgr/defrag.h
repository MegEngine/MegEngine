/**
 * \file src/core/impl/graph/var_node_mem_mgr/defrag.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "../impl_common.h"

namespace mgb {
namespace cg {

/*!
 * \brief defragmenter for device memory used by dynamic variables
 *
 * Currently only enabled for cuda.
 *
 * alloc_var_storage() is thread-safe.
 */
class VarDevMemDefragmenter {
public:
    explicit VarDevMemDefragmenter(VarNodeMemManager* mem_mgr)
            : m_mem_mgr{mem_mgr} {}

private:
    bool m_enable;
    VarNodeMemManager* const m_mem_mgr;

    void alloc_direct(VarNode* var, DeviceTensorStorage& storage, size_t size);

#if MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER
    struct CompNodeInfo {
        std::mutex mtx;
        VarNodeSet vars;
    };
    struct ChunkInfo;

    std::mutex m_mtx;
    CompNode::UnorderedMap<CompNodeInfo> m_cninfo_map;
    ThinHashSet<OperatorNodeBase*> m_move_safe_oprs;

    //! query whether to enable defragmenting for a particular device type
    static bool enable_for_device(CompNode::DeviceType type) {
        return type == CompNode::DeviceType::CUDA;
    }

    //! allocate storage and call defrag() if fails
    void alloc_with_defrag(VarNode* var, DeviceTensorStorage& storage,
                           size_t size);

    /*!
     * \brief perform defragmenting
     *
     * Note: lock must be held before entering this method
     * \param req_var the var that initiates this request
     * \param extra_size size needed to be allocated after defragmenting
     * \return a tensor storage of \p extra_size
     */
    void defrag(VarNode* req_var, const CompNodeInfo& cn_info,
                size_t extra_size);

    void defrag_impl(VarNode* req_var, const CompNodeInfo& cn_info,
                     size_t extra_size);

public:
    /*!
     * \brief allocate storage for a var
     *
     * Defragmenting would be performed if memory allocation fails.
     *
     * \param storage tensor storage associated with the var
     */
    void alloc_var_storage(VarNode* var, DeviceTensorStorage& storage,
                           size_t size) {
        if (!m_enable || !enable_for_device(var->comp_node().device_type())) {
            alloc_direct(var, storage, size);
        } else {
            alloc_with_defrag(var, storage, size);
        }
    }

    //! register a var to be managed by the defragmenter
    void register_var(VarNode* var);

    //! clear all registered vars
    void clear_all();
#else  // MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER
public:
    void alloc_var_storage(VarNode* var, DeviceTensorStorage& storage,
                           size_t size) {
        alloc_direct(var, storage, size);
    }

    void clear_all() {}

    void register_var(VarNode*) {}

#endif  // MGB_ENABLE_VAR_DEV_MEM_DEFRAGMENTER

    //! set whether to enable deragmenting
    void set_enable(bool flag) { m_enable = flag; }
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

