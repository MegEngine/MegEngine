#include "test/naive/rng.h"
#include "megdnn.h"
#include "test/atlas/fixture.h"
#include "test/common/tensor.h"

namespace megdnn {

namespace test {

template <typename ctype>
void assert_uniform_correct(const ctype* src, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        ASSERT_GT(src[i], ctype(0));
        ASSERT_LE(src[i], ctype(1));
    }
    auto stat = get_mean_var(src, size, ctype(0.5));
    ASSERT_LE(std::abs(stat.first - 0.5), 1e-3);
    ASSERT_LE(std::abs(stat.second - 1.0 / 12), 1e-3);
}
template <typename dtype>
void run_uniform(Handle* handle) {
    auto opr = handle->create_operator<UniformRNG>();
    opr->param().dtype = DTypeTrait<dtype>::enumv;
    SyncedTensor<typename DTypeTrait<dtype>::ctype> t(
            handle, {TensorShape{200000}, dtype()});
    opr->exec(t.tensornd_dev(), {});
    assert_uniform_correct(t.ptr_mutable_host(), t.layout().total_nr_elems());
}

template <typename dtype>
void run_gaussian(Handle* handle) {
    using ctype = typename DTypeTrait<dtype>::ctype;
    auto opr = handle->create_operator<GaussianRNG>();
    opr->param().mean = 0.8;
    opr->param().std = 2.3;
    opr->param().dtype = DTypeTrait<dtype>::enumv;
    SyncedTensor<ctype> t(handle, {TensorShape{200001}, dtype()});
    opr->exec(t.tensornd_dev(), {});

    auto ptr = t.ptr_mutable_host();
    auto size = t.layout().total_nr_elems();
    for (size_t i = 0; i < size; ++i) {
        ASSERT_LE(std::abs(ptr[i] - 0.8), ctype(15));
    }
    auto stat = get_mean_var(ptr, size, ctype(0.8));

    ASSERT_LE(std::abs(stat.first - 0.8), 5e-3);
    ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 5e-2);
}

TEST_F(ATLAS, UNIFORM_RNG_F32) {
    run_uniform<dtype::Float32>(handle_atlas());
}

TEST_F(ATLAS, UNIFORM_RNG_F16) {
    DNN_INC_FLOAT16(run_uniform<dtype::Float16>(handle_atlas()));
}

TEST_F(ATLAS, GAUSSIAN_RNG_F32) {
    run_gaussian<dtype::Float32>(handle_atlas());
}

TEST_F(ATLAS, GAUSSIAN_RNG_F16) {
    DNN_INC_FLOAT16(run_gaussian<dtype::Float16>(handle_atlas()));
}

}  // namespace test
}  // namespace megdnn