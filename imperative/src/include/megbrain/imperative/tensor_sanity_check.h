/**
 * \file src/core/include/megbrain/tensor_sanity_check.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/op_def.h"
#include "megbrain/plugin/var_sanity_check.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs/general.h"

namespace mgb {

namespace imperative {

class TensorChecksumCalc {
public:
    using ChecksumResult = megdnn::opr_result::Checksum;
    using Error = VarSanityCheckError;
    struct WorkspaceCache {
        //! var comp node to workspace
        CompNode::UnorderedMap<DeviceTensorStorage> storage;
    };
    ThinHashMap<std::thread::id, WorkspaceCache> m_workspace;
    std::mutex m_workspace_mtx;
    ChecksumResult calc(TensorPtr ptr);
    TensorChecksumCalc() {}
};

class TensorSanityCheckImpl;

class TensorSanityCheck {
public:
    TensorSanityCheck();
    ~TensorSanityCheck();
    void enable();
    void disable();
    std::string print_op(const OpDef& def);
private:
    std::unique_ptr<TensorSanityCheckImpl> m_checker;
};


} // namespace imperative
} // namespace mgb