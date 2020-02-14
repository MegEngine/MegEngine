/**
 * \file dnn/test/cuda/mesh_indexing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "test/common/checker.h"
#include "test/common/index.h"
#include "test/common/mesh_indexing.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, MESH_INDEXING) {
    Checker<MeshIndexing> checker(handle_cuda());
    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    SmallVector<size_t> init_axes;

    idx_size0 = 23;
    init_axes = {0};
    checker.set_proxy({init_axes})
            .execs({{23}, {100}, {100}})
            .execs({{23, 5}, {100, 5}, {100}});

    idx_size0 = 3;
    init_axes = {1};
    checker.set_proxy({init_axes})
            .execs({{2, 3}, {2, 10}, {10}})
            .execs({{2, 3, 5}, {2, 50, 5}, {50}})
            .execs({{2, 3, 5, 7}, {2, 55, 5, 7}, {55}});

    idx_size0 = 23;
    idx_size1 = 17;
    init_axes = {3, 1};
    checker.set_proxy({init_axes})
            .execs({{3, 17, 9, 23}, {3, 100, 9, 100}, {100}, {100}})
            .execs({{3, 17, 29, 30}, {3, 66, 29, 99}, {99}, {66}});
}

TEST_F(CUDA, BATCHED_MESH_INDEXING) {
    Checker<BatchedMeshIndexing> checker(handle_cuda());

    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    SmallVector<size_t> init_axes;

    init_axes = {1};
    idx_size0 = 5;
    checker.set_proxy({init_axes}).execs({{2, 5}, {2, 3}, {2, 3}});

    idx_size0 = 23;
    idx_size1 = 17;
    init_axes = {3, 1};
    checker.set_proxy({init_axes})
            .execs({{3, 17, 9, 23}, {3, 100, 9, 100}, {3, 100}, {3, 100}})
            .execs({{3, 17, 29, 30}, {3, 66, 29, 99}, {3, 99}, {3, 66}});

    idx_size0 = 5;
    init_axes = {1};
    TensorLayout index_layout{TensorShape{1, 3}, dtype::Int32()};
    index_layout = index_layout.broadcast({2, 3});
    checker.set_proxy({init_axes})
            .execl({TensorLayout{TensorShape{2, idx_size0}, dtype::Float32()},
                    TensorLayout{TensorShape{2, 3}, dtype::Float32()},
                    index_layout});
}

namespace {
template <typename T, typename RNG>
void run_modify_test(Handle* handle) {
    Checker<T> checker(handle);
    size_t idx_size0, idx_size1;
    RNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    SmallVector<size_t> init_axes;

    idx_size0 = 230;
    init_axes = {0};
    checker.set_proxy({init_axes})
            .execs({{230}, {100}, {100}})
            .execs({{230, 5}, {100, 5}, {100}});

    idx_size0 = 30;
    init_axes = {1};
    checker.set_proxy(init_axes)
            .execs({{2, 30}, {2, 10}, {10}})
            .execs({{2, 30, 5}, {2, 20, 5}, {20}})
            .execs({{2, 30, 5, 7}, {2, 25, 5, 7}, {25}});
}

template <typename T, typename RNG>
void run_batch_modify_test(Handle* handle) {
    Checker<T> checker(handle);
    size_t idx_size0, idx_size1;
    RNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    SmallVector<size_t> init_axes;

    init_axes = {1};
    idx_size0 = 5;
    checker.set_proxy({init_axes}).execs({{2, 5}, {2, 3}, {2, 3}});

    idx_size0 = 23;
    idx_size1 = 17;
    init_axes = {3, 1};
    checker.set_proxy({init_axes})
            .execs({{3, 17, 9, 23}, {3, 10, 9, 10}, {3, 10}, {3, 10}})
            .execs({{3, 17, 29, 30}, {3, 11, 29, 22}, {3, 22}, {3, 11}});
}
}  // namespace

TEST_F(CUDA, MESH_MODIFY_INCREMENT) {
    run_modify_test<IncrMeshIndexing, IndexRNG>(handle_cuda());
}

TEST_F(CUDA, MESH_MODIFY_SETTING) {
    run_modify_test<SetMeshIndexing, mesh_indexing::NoReplacementIndexRNG>(
            handle_cuda());
}

TEST_F(CUDA, BATCHED_MESH_MODIFY_INCREMENT) {
    run_batch_modify_test<BatchedIncrMeshIndexing, IndexRNG>(handle_cuda());
}

TEST_F(CUDA, BATCHED_MESH_MODIFY_SETTING) {
    run_batch_modify_test<BatchedSetMeshIndexing,
                          mesh_indexing::NoReplacementIndexRNG>(handle_cuda());
}
