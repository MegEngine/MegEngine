/**
 * \file imperative/src/include/megbrain/imperative/ops/io_remote.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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

    size_t hash() const override;
    bool is_same_st(const Hashable& another) const override;

    auto as_tuple() const{
        return std::tuple(key, addr, port, rank_to);
    }
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

    size_t hash() const override;
    bool is_same_st(const Hashable& another) const override;

    auto as_tuple() const{
        return std::tuple(key, addr, port, rank_from, cn, dtype, shape.to_string());
    }
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
