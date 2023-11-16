#include "test/naive/rng.h"
#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/tensor.h"

namespace megdnn {

namespace test {

TEST_F(CAMBRICON, UNIFORM_RNG_F32) {
    auto opr = handle_cambricon()->create_operator<UniformRNG>();
    opr->param().dtype = DTypeTrait<dtype::Float32>::enumv;
    SyncedTensor<> t(handle_cambricon(), {TensorShape{200000}, dtype::Float32()});
    opr->exec(t.tensornd_dev(), {});
    assert_uniform_correct(t.ptr_mutable_host(), t.layout().total_nr_elems());
}

TEST_F(CAMBRICON, GAUSSIAN_RNG_F32) {
    auto opr = handle_cambricon()->create_operator<GaussianRNG>();
    opr->param().mean = 0.8;
    opr->param().std = 2.3;
    opr->param().dtype = DTypeTrait<dtype::Float32>::enumv;
    for (size_t size : {1, 200000, 200001}) {
        TensorLayout ly{{size}, dtype::Float32()};
        Tensor<dt_byte> workspace(
                handle_cambricon(),
                {TensorShape{opr->get_workspace_in_bytes(ly)}, dtype::Byte()});
        SyncedTensor<> t(handle_cambricon(), ly);
        opr->exec(
                t.tensornd_dev(),
                {workspace.ptr(), workspace.layout().total_nr_elems()});

        auto ptr = t.ptr_mutable_host();
        // ASSERT_LE(std::abs(ptr[0] - 0.8), 2.3);

        if (size >= 1000) {
            auto stat = get_mean_var(ptr, size, 0.8f);
            ASSERT_LE(std::abs(stat.first - 0.8), 1e-1);
            ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 1e-1);
        }
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
