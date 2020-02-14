/**
 * \file dnn/test/naive/mesh_indexing.cpp
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
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, MESH_INDEXING) {
    SmallVector<size_t> init_axes;

    auto multi_axis_index_impl = [this,
                                  &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<IndexingMultiAxisVec>();
        OprProxy<IndexingMultiAxisVec> proxy(init_axes);
        proxy.exec(opr.get(), tensors);
    };

    Checker<MeshIndexing> checker(handle());
    checker.set_extra_opr_impl(multi_axis_index_impl);
    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    idx_size0 = 23;
    init_axes = {0};
    checker.set_proxy({init_axes})
            .execs({{23}, {100}, {100}})
            .execs({{23, 5}, {100, 5}, {100}});

    idx_size0 = 3;
    init_axes = {1};
    checker.set_proxy(init_axes)
            .execs({{2, 3}, {2, 10}, {10}})
            .execs({{2, 3, 5}, {2, 50, 5}, {50}})
            .execs({{2, 3, 5, 7}, {2, 55, 5, 7}, {55}});
}

TEST_F(NAIVE, BATCHED_MESH_INDEXING) {
    SmallVector<size_t> init_axes;

    auto extra_impl = [this, &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<MeshIndexing>();
        OprProxy<MeshIndexing> proxy(init_axes);
        size_t N = tensors[0].layout[0];
        for (size_t n = 0; n < N; ++n) {
            TensorNDArray new_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto&& tensor = tensors[i];
                TensorLayout layout = tensor.layout.remove_axis(0);
                if (i < 2) {
                    layout.add_axis_cont_inplace(0);
                }
                void* ptr = static_cast<dt_byte*>(tensor.raw_ptr) +
                            tensor.layout.stride[0] * n *
                                    tensor.layout.dtype.size();
                new_tensors.emplace_back(ptr, layout);
            }
            proxy.exec(opr.get(), new_tensors);
        }
    };

    Checker<BatchedMeshIndexing> checker(handle());
    checker.set_extra_opr_impl(extra_impl);

    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    idx_size0 = 5;
    init_axes = {1};
    checker.set_proxy({init_axes}).execs({{1, idx_size0}, {1, 3}, {1, 3}});

    idx_size0 = 23;
    idx_size1 = 17;
    init_axes = {1, 2};
    checker.set_proxy({init_axes})
            .execs({{7, idx_size0, idx_size1}, {7, 10, 20}, {7, 10}, {7, 20}})
            .execs({{7, idx_size0, idx_size1, 9},
                    {7, 10, 20, 9},
                    {7, 10},
                    {7, 20}});

    init_axes = {2, 1};
    checker.set_proxy({init_axes})
            .execs({{8, idx_size1, idx_size0}, {8, 20, 10}, {8, 10}, {8, 20}})
            .execs({{8, idx_size1, idx_size0, 9},
                    {8, 20, 10, 9},
                    {8, 10},
                    {8, 20}});

    idx_size0 = 5;
    init_axes = {1};
    TensorLayout index_layout{TensorShape{1, 3}, dtype::Int32()};
    index_layout = index_layout.broadcast({2, 3});
    checker.set_proxy({init_axes})
            .execl({TensorLayout{TensorShape{2, idx_size0}, dtype::Float32()},
                    TensorLayout{TensorShape{2, 3}, dtype::Float32()},
                    index_layout});
}

TEST_F(NAIVE, MESH_MODIFY_INCREMENT) {
    SmallVector<size_t> init_axes;

    auto multi_axis_index_impl = [this,
                                  &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<IndexingIncrMultiAxisVec>();
        OprProxy<IndexingIncrMultiAxisVec> proxy(init_axes);
        proxy.exec(opr.get(), tensors);
    };

    Checker<IncrMeshIndexing> checker(handle());
    checker.set_extra_opr_impl(multi_axis_index_impl);
    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    idx_size0 = 23;
    init_axes = {0};
    checker.set_proxy({init_axes})
            .execs({{23}, {100}, {100}})
            .execs({{23, 5}, {100, 5}, {100}});

    idx_size0 = 3;
    init_axes = {1};
    checker.set_proxy(init_axes)
            .execs({{2, 3}, {2, 10}, {10}})
            .execs({{2, 3, 5}, {2, 50, 5}, {50}})
            .execs({{2, 3, 5, 7}, {2, 55, 5, 7}, {55}});
}

TEST_F(NAIVE, BATCHED_MESH_MODIFY_INCREMENT) {
    SmallVector<size_t> init_axes;

    auto extra_impl = [this, &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<IncrMeshIndexing>();
        OprProxy<IncrMeshIndexing> proxy(init_axes);
        size_t N = tensors[0].layout[0];
        for (size_t n = 0; n < N; ++n) {
            TensorNDArray new_tensors;
            for (size_t i = 0; i < tensors.size(); ++i) {
                auto&& tensor = tensors[i];
                TensorLayout layout = tensor.layout.remove_axis(0);
                if (i < 2) {
                    layout.add_axis_cont_inplace(0);
                }
                void* ptr =
                        static_cast<dt_byte*>(tensor.raw_ptr) +
                        tensor.layout.dtype.size(tensor.layout.stride[0] * n);
                new_tensors.emplace_back(ptr, layout);
            }
            proxy.exec(opr.get(), new_tensors);
        }
    };
    Checker<BatchedIncrMeshIndexing> checker(handle());
    checker.set_extra_opr_impl(extra_impl);

    size_t idx_size0, idx_size1;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1);

    idx_size0 = 5;
    init_axes = {1};
    checker.set_proxy({init_axes}).execs({{1, idx_size0}, {1, 3}, {1, 3}});

    idx_size0 = 23;
    idx_size1 = 17;
    init_axes = {1, 2};
    checker.set_proxy({init_axes})
            .execs({{7, idx_size0, idx_size1}, {7, 10, 20}, {7, 10}, {7, 20}})
            .execs({{7, idx_size0, idx_size1, 9},
                    {7, 10, 20, 9},
                    {7, 10},
                    {7, 20}});

    init_axes = {2, 1};
    checker.set_proxy({init_axes})
            .execs({{8, idx_size1, idx_size0}, {8, 20, 10}, {8, 10}, {8, 20}})
            .execs({{8, idx_size1, idx_size0, 9},
                    {8, 20, 10, 9},
                    {8, 10},
                    {8, 20}});
}

TEST_F(NAIVE, MESH_MODIFY_SETTING) {
    SmallVector<size_t> init_axes;

    auto extra_impl = [this, &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<IncrMeshIndexing>();
        OprProxy<IncrMeshIndexing> proxy(init_axes);
        proxy.exec(opr.get(), tensors);
    };
    Checker<SetMeshIndexing> checker(handle());
    checker.set_extra_opr_impl(extra_impl);

    size_t idx_size0, idx_size1;
    mesh_indexing::NoReplacementIndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    ConstValue zero_gen;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1)
            .set_rng(0, &zero_gen);

    idx_size0 = 5;
    init_axes = {1};
    checker.set_proxy({init_axes}).execs({{1, idx_size0}, {1, 3}, {3}});

    idx_size0 = 23;
    idx_size1 = 20;
    init_axes = {1, 2};
    checker.set_proxy({init_axes})
            .execs({{7, idx_size0, idx_size1}, {7, 10, 20}, {10}, {20}})
            .execs({{7, idx_size0, idx_size1, 9}, {7, 10, 20, 9}, {10}, {20}});

    init_axes = {2, 1};
    checker.set_proxy({init_axes})
            .execs({{8, idx_size1, idx_size0}, {8, 20, 10}, {10}, {20}})
            .execs({{8, idx_size1, idx_size0, 9}, {8, 20, 10, 9}, {10}, {20}});
}

TEST_F(NAIVE, BATCHED_MESH_MODIFY_SETTING) {
    SmallVector<size_t> init_axes;

    auto extra_impl = [this, &init_axes](const TensorNDArray& tensors) {
        auto opr = handle()->create_operator<BatchedIncrMeshIndexing>();
        OprProxy<BatchedIncrMeshIndexing> proxy(init_axes);
        proxy.exec(opr.get(), tensors);
    };
    Checker<BatchedSetMeshIndexing> checker(handle());
    checker.set_extra_opr_impl(extra_impl);

    size_t idx_size0, idx_size1;
    mesh_indexing::NoReplacementIndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
    ConstValue zero_gen;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_rng(2, &rng0)
            .set_rng(3, &rng1)
            .set_rng(0, &zero_gen);

    idx_size0 = 5;
    init_axes = {1};
    checker.set_proxy({init_axes}).execs({{1, idx_size0}, {1, 3}, {1, 3}});

    idx_size0 = 23;
    idx_size1 = 20;
    init_axes = {1, 2};
    checker.set_proxy({init_axes})
            .execs({{7, idx_size0, idx_size1}, {7, 10, 20}, {7, 10}, {7, 20}})
            .execs({{7, idx_size0, idx_size1, 9},
                    {7, 10, 20, 9},
                    {7, 10},
                    {7, 20}});

    init_axes = {2, 1};
    checker.set_proxy({init_axes})
            .execs({{8, idx_size1, idx_size0}, {8, 20, 10}, {8, 10}, {8, 20}})
            .execs({{8, idx_size1, idx_size0, 9},
                    {8, 20, 10, 9},
                    {8, 10},
                    {8, 20}});
}
