#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

namespace {
class ArgsortRNG final : public RNG {
    bool m_rev_order = false;
    DType m_dtype;

    template <typename T>
    void fill(T* ptr, int n) {
        if (m_rev_order) {
            for (int i = 0; i < n; ++i) {
                ptr[i] = static_cast<T>(n / 2 - i);
            }
        } else {
            for (int i = 0; i < n; ++i)
                ptr[i] = static_cast<T>(i - n / 2);
            COMPAT_RANDOM(ptr, ptr + n);
        }
    }

    void gen(const TensorND& tensor) override {
        auto n = tensor.layout.total_nr_elems();
        if (m_dtype == dtype::Float32{}) {
            fill(tensor.ptr<dt_float32>(), n);
        } else {
            megdnn_assert(m_dtype == dtype::Int32{});
            fill(tensor.ptr<dt_int32>(), n);
        }
    }

public:
    ArgsortRNG(DType dt) : m_dtype{dt} {}

    void set_rev_order(bool flag) { m_rev_order = flag; }
};

void run_forward_test(Handle* handle, DType dtype) {
    Checker<ArgsortForward> checker(handle);
    using Param = Argsort::Param;
    using Order = Param::Order;
    ArgsortRNG rng{dtype};
    checker.set_dtype(2, dtype::Int32());
    checker.set_dtype(0, dtype).set_rng(0, &rng);
    for (size_t i = 3; i < 10240; i *= 2) {
        Param param;
        param.order = Order::ASCENDING;
        checker.set_param(param).execs({{1, i + 1}, {}, {}});
        checker.set_param(param).execs({{3, i + 1}, {}, {}});
        param.order = Order::DESCENDING;
        checker.set_param(param).execs({{4, i - 1}, {}, {}});
        checker.set_param(param).execs({{13, i + 3}, {}, {}});
    }
}

}  // anonymous namespace

TEST_F(ATLAS, ARGSORT_FORWARD_F32) {
    run_forward_test(handle_atlas(), dtype::Float32{});
}

TEST_F(ATLAS, ARGSORT_FORWARD_I32) {
    run_forward_test(handle_atlas(), dtype::Int32{});
}

// vim: syntax=cpp.doxygen
