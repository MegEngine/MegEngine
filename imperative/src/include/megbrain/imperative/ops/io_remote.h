/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

class RemoteSend : public OpDefImplBase<RemoteSend> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    RemoteSend() = default;
    RemoteSend(const std::string& key_, const std::string& addr_,
               uint32_t port_, uint32_t rank_to_)
            : key(key_),
              addr(addr_),
              port(port_),
              rank_to(rank_to_) {}
    std::string key;
    std::string addr;
    uint32_t port;
    uint32_t rank_to;
};

class RemoteRecv : public OpDefImplBase<RemoteRecv> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    RemoteRecv() = default;
    RemoteRecv(const std::string& key_, const std::string& addr_,
               uint32_t port_, uint32_t rank_from_, TensorShape shape_,
               CompNode cn_, const DType& dtype_)
            : key(key_),
              addr(addr_),
              port(port_),
              rank_from(rank_from_),
              cn(cn_),
              shape(shape_),
              dtype(dtype_) {}
    std::string key;
    std::string addr;
    uint32_t port;
    uint32_t rank_from;
    CompNode cn;
    TensorShape shape;
    DType dtype;
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
