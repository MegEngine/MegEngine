/**
 * \file dnn/test/common/exec_proxy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"

#include "test/common/workspace_wrapper.h"

#include <cstddef>
#include <vector>

namespace megdnn {
namespace test {

template <typename Opr, size_t Arity, bool has_workspace>
struct ExecProxy;

template <typename Opr>
struct ExecProxy<Opr, 8, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                tensors[3].layout, tensors[4].layout, tensors[5].layout,
                tensors[6].layout, tensors[7].layout));
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  tensors[5], tensors[6], tensors[7], W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 6, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                tensors[3].layout, tensors[4].layout, tensors[5].layout));
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  tensors[5], W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 5, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                tensors[3].layout, tensors[4].layout));
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 4, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                tensors[3].layout));
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3],
                  W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 3, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout));
        opr->exec(tensors[0], tensors[1], tensors[2], W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 2, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(tensors[0].layout,
                                             tensors[1].layout));
        opr->exec(tensors[0], tensors[1], W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 1, true> {
    WorkspaceWrapper W;
    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(tensors[0].layout));
        opr->exec(tensors[0], W.workspace());
    }
};

template <typename Opr>
struct ExecProxy<Opr, 5, false> {
    void exec(Opr* opr, const TensorNDArray& tensors) {
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]);
    }
};

template <typename Opr>
struct ExecProxy<Opr, 4, false> {
    void exec(Opr* opr, const TensorNDArray& tensors) {
        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3]);
    }
};

template <typename Opr>
struct ExecProxy<Opr, 3, false> {
    void exec(Opr* opr, const TensorNDArray& tensors) {
        opr->exec(tensors[0], tensors[1], tensors[2]);
    }
};

template <typename Opr>
struct ExecProxy<Opr, 2, false> {
    void exec(Opr* opr, const TensorNDArray& tensors) {
        opr->exec(tensors[0], tensors[1]);
    }
};

template <typename Opr>
struct ExecProxy<Opr, 7, true> {
    WorkspaceWrapper W;

    void exec(Opr* opr, const TensorNDArray& tensors) {
        if (!W.valid()) {
            W = WorkspaceWrapper(opr->handle(), 0);
        }
        W.update(opr->get_workspace_in_bytes(
                tensors[0].layout, tensors[1].layout, tensors[2].layout,
                tensors[3].layout, tensors[4].layout, tensors[5].layout,
                tensors[6].layout));

        opr->exec(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                  tensors[5], tensors[6], W.workspace());
    }
};

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
