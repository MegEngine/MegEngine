/**
 * \file dnn/test/cuda/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/rng.h"
#include "megdnn/oprs.h"
#include "test/common/tensor.h"
#include "test/cuda/fixture.h"

namespace megdnn {

namespace test {

namespace {

template <typename T>
void run_gamma(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto opr = handle->create_operator<GammaRNG>();

    TensorLayout ly{TensorShape{2000000 * 5}, T()};

    SyncedTensor<ctype> out(handle, ly);
    SyncedTensor<ctype> shape(handle, ly);
    SyncedTensor<ctype> scale(handle, ly);
    auto shape_ptr = shape.ptr_mutable_host();
    auto scale_ptr = scale.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2000000; ++j) {
            shape_ptr[i * 2000000 + j] = 2 * 0.3 * i + 0.3;
            scale_ptr[i * 2000000 + j] = i * 0.2 + 0.1;
        }
    }

    opr->exec(shape.tensornd_dev(), scale.tensornd_dev(), out.tensornd_dev(), {});

    auto ptr = out.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        float a = 2 * 0.3 * i + 0.3, b = i * 0.2 + 0.1;
        float mean = a * b;
        float std = a * (b * b);
        auto stat = get_mean_var(ptr + i * 2000000, 2000000, ctype(mean));
        ASSERT_LE(std::abs(stat.first - mean), 0.01);
        ASSERT_LE(std::abs(stat.second - std), 0.01);
    }
}

template <typename T>
void run_poisson(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto opr = handle->create_operator<PoissonRNG>();

    TensorLayout ly{TensorShape{200000 * 5}, T()};

    SyncedTensor<ctype> out(handle, ly);
    SyncedTensor<ctype> lam(handle, ly);
    auto lam_ptr = lam.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 200000; ++j) {
            lam_ptr[i * 200000 + j] = ctype(i + 1);
        }
    }
    opr->exec(lam.tensornd_dev(), out.tensornd_dev(), {});

    auto ptr = out.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        auto stat = get_mean_var(ptr + i * 200000, 200000, ctype(i + 1));
        ASSERT_LE(std::abs(stat.first - ctype(i + 1)), 0.01);
        ASSERT_LE(std::abs(stat.second - ctype(i + 1)), 0.01);
    }
}

template <typename T>
void run_beta(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto opr = handle->create_operator<BetaRNG>();

    TensorLayout ly{TensorShape{200000 * 5}, T()};

    SyncedTensor<ctype> out(handle, ly);
    SyncedTensor<ctype> alpha(handle, ly);
    SyncedTensor<ctype> beta(handle, ly);
    auto alpha_ptr = alpha.ptr_mutable_host();
    auto beta_ptr = beta.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 200000; ++j) {
            alpha_ptr[i * 200000 + j] = 0.3 * i + 0.1;
            beta_ptr[i * 200000 + j] = 2 * i * 0.3 + 0.1;
        }
    }

    opr->exec(alpha.tensornd_dev(), beta.tensornd_dev(), out.tensornd_dev(), {});

    auto ptr = out.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        float a = 0.3 * i + 0.1, b = 2 * i * 0.3 + 0.1;
        float mean = a / (a + b);
        float std = a * b / ((a + b) * (a + b) * (a + b + 1));
        auto stat = get_mean_var(ptr + i * 200000, 200000, ctype(mean));
        ASSERT_LE(std::abs(stat.first - mean), 0.01);
        ASSERT_LE(std::abs(stat.second - std), 0.01);
    }
}

template <typename T>
void run_permutation(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    size_t sample_num = std::min(200000, static_cast<int>(DTypeTrait<T>::max()) - 10);

    auto opr = handle->create_operator<PermutationRNG>();
    opr->param().dtype = DTypeTrait<T>::enumv;
    TensorLayout ly{TensorShape{sample_num}, T()};
    Tensor<dt_byte> workspace(
            handle, {TensorShape{opr->get_workspace_in_bytes(ly)}, dtype::Byte()});
    SyncedTensor<ctype> t(handle, ly);

    opr->exec(t.tensornd_dev(), {workspace.ptr(), workspace.layout().total_nr_elems()});

    auto ptr = t.ptr_mutable_host();
    auto size = t.layout().total_nr_elems();

    std::vector<ctype> res(size);
    int not_same = 0;
    for (size_t i = 0; i < size; ++i) {
        if ((ptr[i] - ctype(i)) >= ctype(1))
            not_same++;
        res[i] = ptr[i];
    }
    ASSERT_GT(not_same, 5000);
    std::sort(res.begin(), res.end());
    for (size_t i = 0; i < size; ++i) {
        ASSERT_LE(std::abs(res[i] - ctype(i)), 1e-8);
    }
}

template <typename T>
void run_shuffle(Handle* handle, bool bwd_flag) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto run = [&](TensorShape shape) {
        auto opr = handle->create_operator<ShuffleRNGForward>();
        TensorLayout srclay{shape, T()};
        TensorLayout dstlay{shape, T()};
        TensorLayout indexlay{TensorShape{shape[0]}, dtype::Int32()};
        Tensor<dt_byte> workspace(
                handle,
                {TensorShape{opr->get_workspace_in_bytes(srclay, dstlay, indexlay)},
                 dtype::Byte()});
        SyncedTensor<ctype> src(handle, srclay);
        SyncedTensor<ctype> dst(handle, dstlay);
        SyncedTensor<DTypeTrait<dt_int32>::ctype> index(handle, indexlay);
        auto sptr = src.ptr_mutable_host();
        size_t size = src.layout().total_nr_elems();
        for (size_t j = 0; j < size; ++j) {
            sptr[j] = j;
        }
        opr->exec(
                src.tensornd_dev(), dst.tensornd_dev(), index.tensornd_dev(),
                {workspace.ptr(), workspace.layout().total_nr_elems()});

        auto dptr = dst.ptr_mutable_host();
        auto iptr = index.ptr_mutable_host();
        size_t len = index.layout().total_nr_elems();
        size_t step = size / len;
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < step; ++j) {
                ASSERT_EQ(dptr[i * step + j], sptr[iptr[i] * step + j]);
            }
        }
        if (bwd_flag) {
            for (size_t j = 0; j < size; ++j) {
                sptr[j] = 0;
            }
            auto oprbwd = handle->create_operator<ShuffleRNGBackward>();
            oprbwd->exec(
                    dst.tensornd_dev(), index.tensornd_dev(), src.tensornd_dev(),
                    {workspace.ptr(), workspace.layout().total_nr_elems()});
            auto sptr_bwd = src.ptr_mutable_host();
            for (size_t i = 0; i < len; ++i) {
                for (size_t j = 0; j < step; ++j) {
                    ASSERT_EQ(dptr[i * step + j], sptr_bwd[iptr[i] * step + j]);
                }
            }
        }
    };

    run({10});
    run({6, 3});
}

}  // anonymous namespace

TEST_F(CUDA, UNIFORM_RNG_F32) {
    auto opr = handle_cuda()->create_operator<UniformRNG>();
    opr->param().dtype = DTypeTrait<dtype::Float32>::enumv;
    SyncedTensor<> t(handle_cuda(), {TensorShape{200000}, dtype::Float32()});
    opr->exec(t.tensornd_dev(), {});

    assert_uniform_correct(t.ptr_mutable_host(), t.layout().total_nr_elems());
}

TEST_F(CUDA, GAUSSIAN_RNG_F32) {
    auto opr = handle_cuda()->create_operator<GaussianRNG>();
    opr->param().mean = 0.8;
    opr->param().std = 2.3;
    opr->param().dtype = DTypeTrait<dtype::Float32>::enumv;
    for (size_t size : {1, 200000, 200001}) {
        TensorLayout ly{{size}, dtype::Float32()};
        Tensor<dt_byte> workspace(
                handle_cuda(),
                {TensorShape{opr->get_workspace_in_bytes(ly)}, dtype::Byte()});
        SyncedTensor<> t(handle_cuda(), ly);
        opr->exec(
                t.tensornd_dev(),
                {workspace.ptr(), workspace.layout().total_nr_elems()});

        auto ptr = t.ptr_mutable_host();
        ASSERT_LE(std::abs(ptr[0] - 0.8), 2.3);

        if (size >= 1000) {
            auto stat = get_mean_var(ptr, size, 0.8f);
            ASSERT_LE(std::abs(stat.first - 0.8), 5e-3);
            ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 5e-2);
        }
    }
}

TEST_F(CUDA, GAMMA_RNG_F32) {
    run_gamma<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, GAMMA_RNG_F16) {
    run_gamma<dtype::Float16>(handle_cuda());
}

TEST_F(CUDA, POISSON_RNG_F32) {
    run_poisson<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, POISSON_RNG_F16) {
    run_poisson<dtype::Float16>(handle_cuda());
}

TEST_F(CUDA, BETA_RNG_F32) {
    run_beta<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, BETA_RNG_F16) {
    run_beta<dtype::Float16>(handle_cuda());
}

TEST_F(CUDA, PERMUTATION_RNG_F32) {
    run_permutation<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, PERMUTATION_RNG_INT32) {
    run_permutation<dtype::Int32>(handle_cuda());
}

TEST_F(CUDA, PERMUTATION_RNG_INT16) {
    run_permutation<dtype::Int16>(handle_cuda());
}

TEST_F(CUDA, SHUFFLE_RNG_F32) {
    run_shuffle<dtype::Float32>(handle_cuda(), false);
}

TEST_F(CUDA, SHUFFLE_RNG_INT32) {
    run_shuffle<dtype::Int32>(handle_cuda(), false);
}

TEST_F(CUDA, SHUFFLE_RNG_F16) {
    run_shuffle<dtype::Float16>(handle_cuda(), false);
}

TEST_F(CUDA, SHUFFLE_RNG_BWD_F32) {
    run_shuffle<dtype::Float32>(handle_cuda(), true);
}

TEST_F(CUDA, SHUFFLE_RNG_BWD_INT32) {
    run_shuffle<dtype::Int32>(handle_cuda(), true);
}

TEST_F(CUDA, SHUFFLE_RNG_BWD_F16) {
    run_shuffle<dtype::Float16>(handle_cuda(), true);
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
