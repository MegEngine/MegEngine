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
void run_multinomial(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto opr = handle->create_operator<MultinomialRNG>();

    size_t num_groups = 2;
    size_t num_samples = 10000;
    size_t len_probs = 4;
    bool replacement = true;

    TensorLayout ly_out{TensorShape{num_groups, num_samples}, dtype::Int32()};
    TensorLayout ly_probs{TensorShape{num_groups, len_probs}, T()};
    SyncedTensor<dt_int32> out(handle, ly_out);
    SyncedTensor<ctype> probs(handle, ly_probs);

    auto probs_ptr = probs.ptr_mutable_host();
    probs_ptr[0] = 0.1;
    probs_ptr[1] = 0.2;
    probs_ptr[2] = 0.3;
    probs_ptr[3] = 0.4;
    probs_ptr[4] = 0.0;
    probs_ptr[5] = 0.7;
    probs_ptr[6] = 0.2;
    probs_ptr[7] = 0.1;

    Tensor<dt_byte> workspace(
            handle, {TensorShape{opr->get_workspace_in_bytes(ly_probs, ly_out)},
                     dtype::Byte()});

    opr->param().num_samples = num_samples;
    opr->param().replacement = replacement;
    opr->exec(
            probs.tensornd_dev(), out.tensornd_dev(),
            {workspace.ptr(), workspace.layout().total_nr_elems()});

    auto ptr = out.ptr_mutable_host();

    std::vector<float> sample_probs(num_groups * len_probs, 0);
    for (size_t i = 0; i < num_groups; ++i) {
        for (size_t j = 0; j < num_samples; ++j) {
            sample_probs[i * len_probs + ptr[i * num_samples + j]] += 1;
        }
    }
    for (size_t i = 0; i < num_groups * len_probs; ++i) {
        sample_probs[i] /= num_samples;
    }

    for (size_t i = 0; i < num_groups * len_probs; ++i) {
        ASSERT_LE(std::abs(sample_probs[i] - probs_ptr[i]), 1e-2);
    }

    std::vector<float> float_data_group0;
    std::vector<float> float_data_group1;
    for (size_t i = 0; i < num_samples; ++i) {
        float_data_group0.push_back(static_cast<float>(ptr[i]));
    }
    for (size_t i = num_samples; i < 2 * num_samples; ++i) {
        float_data_group1.push_back(static_cast<float>(ptr[i]));
    }
    float compare_mean_group0 = 0 * 0.1 + 1 * 0.2 + 2 * 0.3 + 3 * 0.4;
    float compare_mean_group1 = 0 * 0.0 + 1 * 0.7 + 2 * 0.2 + 3 * 0.1;
    float compare_var_group0 = (0 * 0.1 + 1 * 0.2 + 4 * 0.3 + 9 * 0.4) -
                               compare_mean_group0 * compare_mean_group0;
    float compare_var_group1 = (0 * 0.0 + 1 * 0.7 + 4 * 0.2 + 9 * 0.1) -
                               compare_mean_group1 * compare_mean_group1;
    auto stat_group0 =
            get_mean_var(float_data_group0.data(), num_samples, compare_mean_group0);
    auto stat_group1 =
            get_mean_var(float_data_group1.data(), num_samples, compare_mean_group1);
    ASSERT_LE(
            std::abs(stat_group0.first - compare_mean_group0),
            compare_mean_group0 * 1e-2);
    ASSERT_LE(
            std::abs(stat_group1.first - compare_mean_group1),
            compare_mean_group1 * 1e-2);
    ASSERT_LE(
            std::abs(stat_group0.second - compare_var_group0),
            compare_var_group0 * 3e-2);
    ASSERT_LE(
            std::abs(stat_group1.second - compare_var_group1),
            compare_var_group1 * 3e-2);
}

template <typename T>
void run_multinomial_without_replacement(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;

    size_t num_groups = 2;
    size_t num_samples = 1;
    size_t len_probs = 4;
    bool replacement = false;
    size_t total_count = 10000;

    TensorLayout ly_probs{TensorShape{num_groups, len_probs}, T()};
    SyncedTensor<ctype> probs(handle, ly_probs);
    auto probs_ptr = probs.ptr_mutable_host();
    probs_ptr[0] = 1;
    probs_ptr[1] = 2;
    probs_ptr[2] = 3;
    probs_ptr[3] = 4;
    probs_ptr[4] = 0;
    probs_ptr[5] = 7;
    probs_ptr[6] = 2;
    probs_ptr[7] = 1;

    std::vector<float> norm_probs;
    for (size_t i = 0; i < 8; ++i) {
        norm_probs.push_back(probs_ptr[i] / 10);
    }

    auto opr = handle->create_operator<MultinomialRNG>();
    opr->param().num_samples = num_samples;
    opr->param().replacement = replacement;
    TensorLayout ly_out{TensorShape{num_groups, num_samples}, dtype::Int32()};
    SyncedTensor<dt_int32> out(handle, ly_out);
    Tensor<dt_byte> workspace(
            handle, {TensorShape{opr->get_workspace_in_bytes(ly_probs, ly_out)},
                     dtype::Byte()});
    std::vector<float> sample_probs(num_groups * len_probs, 0);
    for (size_t i = 0; i < total_count; ++i) {
        opr->exec(
                probs.tensornd_dev(), out.tensornd_dev(),
                {workspace.ptr(), workspace.layout().total_nr_elems()});

        auto ptr = out.ptr_mutable_host();
        sample_probs[ptr[0]] += 1;
        sample_probs[len_probs + ptr[1]] += 1;
    }

    for (size_t i = 0; i < num_groups * len_probs; ++i) {
        sample_probs[i] /= total_count * num_samples;
    }

    for (size_t i = 0; i < num_groups * len_probs; ++i) {
        ASSERT_LE(std::abs(sample_probs[i] - norm_probs[i]), 1e-2);
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

template <typename T>
void run_exponential(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto opr = handle->create_operator<ExponentialRNG>();

    TensorLayout ly{TensorShape{200000 * 5}, T()};

    SyncedTensor<ctype> out(handle, ly);
    SyncedTensor<ctype> rate(handle, ly);
    auto rate_ptr = rate.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 200000; ++j) {
            rate_ptr[i * 200000 + j] = ctype(i + 1);
        }
    }
    opr->exec(rate.tensornd_dev(), out.tensornd_dev(), {});

    auto ptr = out.ptr_mutable_host();
    for (int i = 0; i < 5; ++i) {
        auto stat = get_mean_var(ptr + i * 200000, 200000, ctype(i + 1));
        float mean = 1.0 / (i + 1);
        float var = 1.0 / ((i + 1) * (i + 1));
        ASSERT_LE(std::abs(stat.first - mean), 0.01);
        ASSERT_LE(std::abs(stat.second - var), 0.01);
    }
}

template <typename T>
void run_dropout(Handle* handle) {
    using ctype = typename DTypeTrait<T>::ctype;
    auto run = [&](TensorShape shape, float drop_prob) {
        auto fwd = handle->create_operator<DropoutForward>();
        auto bwd = handle->create_operator<DropoutBackward>();
        fwd->param().drop_prob = drop_prob;
        bwd->param().drop_prob = drop_prob;
        double scale = 1.0 / (1.0 - drop_prob);

        TensorLayout inp_lay{shape, T()};
        TensorLayout oup_lay{shape, T()};
        TensorLayout mask_lay{{fwd->get_mask_size_in_bytes(inp_lay)}, dtype::Byte()};
        TensorLayout doup_lay{shape, T()};
        TensorLayout dinp_lay{shape, T()};
        TensorLayout fwd_ws_lay{
                {fwd->get_workspace_in_bytes(inp_lay, oup_lay, mask_lay)},
                dtype::Byte()};
        TensorLayout bwd_ws_lay{
                {bwd->get_workspace_in_bytes(doup_lay, mask_lay, dinp_lay)},
                dtype::Byte()};

        SyncedTensor<ctype> inp(handle, inp_lay);
        SyncedTensor<ctype> oup(handle, oup_lay);
        SyncedTensor<DTypeTrait<dt_byte>::ctype> mask(handle, mask_lay);
        SyncedTensor<ctype> doup(handle, doup_lay);
        SyncedTensor<ctype> dinp(handle, dinp_lay);
        SyncedTensor<DTypeTrait<dt_byte>::ctype> fwd_ws(handle, fwd_ws_lay);
        SyncedTensor<DTypeTrait<dt_byte>::ctype> bwd_ws(handle, bwd_ws_lay);

        for (size_t i = 0; i < inp.layout().total_nr_elems(); ++i) {
            inp.ptr_mutable_host()[i] = 1;
            doup.ptr_mutable_host()[i] = 1;
        }

        fwd->exec(
                inp.tensornd_dev(), oup.tensornd_dev(), mask.tensornd_dev(),
                {fwd_ws.ptr_mutable_dev(), fwd_ws.layout().total_nr_elems()});
        size_t droped_cnt = 0;
        for (size_t i = 0; i < inp.layout().total_nr_elems(); ++i) {
            ASSERT_TRUE(
                    oup.ptr_host()[i] == 0 ||
                    oup.ptr_host()[i] == static_cast<ctype>(scale));
            if (oup.ptr_host()[i] == 0) {
                droped_cnt++;
            }
        }
        float real_drop = droped_cnt * 1.0 / inp.layout().total_nr_elems();
        ASSERT_LT(abs(drop_prob - real_drop), 1e-2);

#if CUDNN_VERSION >= 7000
        bwd->exec(
                doup.tensornd_dev(), mask.tensornd_dev(), dinp.tensornd_dev(),
                {bwd_ws.ptr_mutable_dev(), bwd_ws.layout().total_nr_elems()});
        for (size_t i = 0; i < inp.layout().total_nr_elems(); ++i) {
            ASSERT_TRUE(oup.ptr_host()[i] == dinp.ptr_host()[i]);
        }
#endif
    };

    run({32, 32, 32, 32}, 0.2);
    run({100000}, 0.3);
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

TEST_F(CUDA, MULTINOMIAL_RNG_F32) {
    run_multinomial<dtype::Float32>(handle_cuda());
    run_multinomial_without_replacement<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, MULTINOMIAL_RNG_F16) {
    run_multinomial<dtype::Float16>(handle_cuda());
    run_multinomial_without_replacement<dtype::Float16>(handle_cuda());
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

TEST_F(CUDA, DROPOUT_F32) {
    run_dropout<dtype::Float32>(handle_cuda());
}

TEST_F(CUDA, DROPOUT_F16) {
    run_dropout<dtype::Float16>(handle_cuda());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
