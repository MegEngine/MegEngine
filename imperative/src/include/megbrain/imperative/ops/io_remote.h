#pragma once
#include "megbrain/imperative/op_def.h"
namespace mgb::imperative {
struct BatchSendRecvOp final : OpDefImplBase<BatchSendRecvOp> {
    SmallVector<std::shared_ptr<OpDef>> op_list;
    BatchSendRecvOp() = default;
    BatchSendRecvOp(SmallVector<std::shared_ptr<OpDef>> op_list) : op_list{op_list} {}
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
};

}  // namespace mgb::imperative