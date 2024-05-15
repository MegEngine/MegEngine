#include "test/common/relayout.h"
#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class ATLAS_RELAYOUT : public ATLAS {};
TYPED_TEST_SUITE(ATLAS_RELAYOUT, relayout::test_types);
TYPED_TEST(ATLAS_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle_atlas());
}
}  // namespace

TEST_F(ATLAS, RELAYOUT_TRANSPOSE) {
    Checker<Relayout> checker(handle_atlas());
    auto run = [&](size_t batch, size_t m, size_t n, size_t c, DType dtype) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype);
        TensorLayout src = {{batch, m, n, c}, dtype};
        src.stride[0] = m * n * c;
        src.stride[1] = c;
        src.stride[2] = m * c;
        src.stride[3] = 1;
        TensorLayout dst = {{batch, m, n, c}, dtype};
        dst.init_contiguous_stride();
        checker.execl({src, dst});
    };
    run(16, 30, 40, 4, dtype::Int8());
    run(16, 20, 10, 4, dtype::Int8());
    run(1, 30, 20, 1, dtype::Int32());
    run(1, 20, 30, 1, dtype::Int32());
    run(1, 11, 21, 1, dtype::Float32());
}

TEST_F(ATLAS, RELAYOUT) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    {
        // contiguous stride
        args.emplace_back(
                TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Float16()),
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Float16()));
        args.emplace_back(
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Float16()),
                TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Float16()));
        args.emplace_back(
                TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype::Float16()),
                TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype::Float16()));
    }
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float16()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Float16()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({16, 128, 128}, {49152, 384, 3}, dtype::Float32()),
            TensorLayout({16, 128, 128}, {16384, 128, 1}, dtype::Float32()));

    {
        // 1d
        size_t n = 10000;
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Int32()),
                TensorLayout({n}, {1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Int32()),
                TensorLayout({n}, {2}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Int32()),
                TensorLayout({n}, {1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Int32()),
                TensorLayout({n}, {2}, dtype::Int32()));
    }
    {
        // 2d
        size_t m = 200, n = 300, k = 400;
        ptrdiff_t k2 = k * 2;
        args.emplace_back(
                TensorLayout({m, n}, {k2, 2}, dtype::Int32()),
                TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {2, k2}, dtype::Int32()),
                TensorLayout({m, n}, {2, k2 + 1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {2, k2}, dtype::Int32()),
                TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 2}, dtype::Int32()),
                TensorLayout({m, n}, {2, k2 + 1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 1}, dtype::Int32()),
                TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {1, k2}, dtype::Int32()),
                TensorLayout({m, n}, {1, k2 + 1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {1, k2}, dtype::Int32()),
                TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int32()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 1}, dtype::Int32()),
                TensorLayout({m, n}, {1, k2 + 1}, dtype::Int32()));
    }
    {
        // 3d
        size_t m = 20, n = 30, k = 40;
        ptrdiff_t k2 = k;
        args.emplace_back(
                TensorLayout({m, n, k}, {k2 * k2 * 4, k2 * 3, 2}, dtype::Int32()),
                TensorLayout(
                        {m, n, k}, {2 * k2 * k2 * k2 * 4, k2 * 3, 2}, dtype::Int32()));
    }
    {
        // simplify_layout
        // 234..56
        // 2..3456
        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int32()),
                TensorLayout(
                        {2, 3, 4, 5, 6}, {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                        dtype::Int32()));
    }

    Checker<Relayout> checker(handle_atlas());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(ATLAS, RELAYOUT_INT8) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    {
        // contiguous stride
        args.emplace_back(
                TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Int8()),
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Int8()),
                TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype::Int8()),
                TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype::Int8()));
    }
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({16, 128, 128}, {49152, 384, 3}, dtype::Int8()),
            TensorLayout({16, 128, 128}, {16384, 128, 1}, dtype::Int8()));

    {
        // 1d
        size_t n = 10000;
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Int8()),
                TensorLayout({n}, {1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Int8()),
                TensorLayout({n}, {2}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Int8()),
                TensorLayout({n}, {1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Int8()),
                TensorLayout({n}, {2}, dtype::Int8()));
    }
    {
        // 2d
        size_t m = 200, n = 300, k = 400;
        ptrdiff_t k2 = k * 2;
        args.emplace_back(
                TensorLayout({m, n}, {k2, 2}, dtype::Int8()),
                TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {2, k2}, dtype::Int8()),
                TensorLayout({m, n}, {2, k2 + 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {2, k2}, dtype::Int8()),
                TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 2}, dtype::Int8()),
                TensorLayout({m, n}, {2, k2 + 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 1}, dtype::Int8()),
                TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {1, k2}, dtype::Int8()),
                TensorLayout({m, n}, {1, k2 + 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {1, k2}, dtype::Int8()),
                TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({m, n}, {k2, 1}, dtype::Int8()),
                TensorLayout({m, n}, {1, k2 + 1}, dtype::Int8()));
    }
    {
        // 3d
        size_t m = 20, n = 30, k = 40;
        ptrdiff_t k2 = k;
        args.emplace_back(
                TensorLayout({m, n, k}, {k2 * k2 * 4, k2 * 3, 2}, dtype::Int8()),
                TensorLayout(
                        {m, n, k}, {2 * k2 * k2 * k2 * 4, k2 * 3, 2}, dtype::Int8()));
    }
    {
        // simplify_layout
        // 234..56
        // 2..3456
        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int8()),
                TensorLayout(
                        {2, 3, 4, 5, 6}, {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                        dtype::Int8()));

        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {4 * 3 * 4 * 5 * 6, 4 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int8()),
                TensorLayout(
                        {2, 3, 4, 5, 6}, {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                        dtype::Int8()));
    }

    Checker<Relayout> checker(handle_atlas());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(ATLAS, RELAYOUT_TEST) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    //! dst contig
    args.emplace_back(
            TensorLayout({5, 32, 9}, {288, 1, 32}, dtype::Int8()),
            TensorLayout({5, 9, 32}, {288, 32, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 9, 32}, {288, 1, 9}, dtype::Int8()),
            TensorLayout({5, 32, 9}, {288, 9, 1}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 4, 9}, {36, 1, 4}, dtype::Int8()),
            TensorLayout({5, 9, 4}, {36, 4, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 9, 4}, {36, 1, 9}, dtype::Int8()),
            TensorLayout({5, 4, 9}, {36, 9, 1}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 32, 4}, {128, 1, 32}, dtype::Int8()),
            TensorLayout({5, 4, 32}, {128, 32, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 4, 32}, {128, 1, 4}, dtype::Int8()),
            TensorLayout({5, 32, 4}, {128, 4, 1}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 7, 5}, {35, 1, 7}, dtype::Int8()),
            TensorLayout({5, 5, 7}, {35, 7, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 5, 7}, {35, 1, 5}, dtype::Int8()),
            TensorLayout({5, 7, 5}, {35, 5, 1}, dtype::Int8()));
    //! src contig
    args.emplace_back(
            TensorLayout({5, 9, 32}, {288, 32, 1}, dtype::Int8()),
            TensorLayout({5, 32, 9}, {288, 1, 32}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 32, 9}, {288, 9, 1}, dtype::Int8()),
            TensorLayout({5, 9, 32}, {288, 1, 9}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 9, 4}, {36, 4, 1}, dtype::Int8()),
            TensorLayout({5, 4, 9}, {36, 1, 4}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 4, 9}, {36, 9, 1}, dtype::Int8()),
            TensorLayout({5, 9, 4}, {36, 1, 9}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 4, 32}, {128, 32, 1}, dtype::Int8()),
            TensorLayout({5, 32, 4}, {128, 1, 32}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 32, 4}, {128, 4, 1}, dtype::Int8()),
            TensorLayout({5, 4, 32}, {128, 1, 4}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 5, 7}, {35, 7, 1}, dtype::Int8()),
            TensorLayout({5, 7, 5}, {35, 1, 7}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 7, 5}, {35, 5, 1}, dtype::Int8()),
            TensorLayout({5, 5, 7}, {35, 1, 5}, dtype::Int8()));
    //! cross
    args.emplace_back(
            TensorLayout({5, 9, 32}, {288 * 4, 32 * 3, 1}, dtype::Int8()),
            TensorLayout({5, 32, 9}, {288 * 4, 1, 32 * 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 32, 9}, {288 * 3, 9 * 2, 1}, dtype::Int8()),
            TensorLayout({5, 9, 32}, {288 * 3, 1, 9 * 2}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 9, 4}, {36 * 10, 4 * 7, 1}, dtype::Int8()),
            TensorLayout({5, 4, 9}, {36 * 10, 1, 4 * 7}, dtype::Int8()));

    Checker<Relayout> checker(handle_atlas());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(ATLAS, RELAYOUT_SRC_NEG_STRIDE) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    args.emplace_back(
            TensorLayout({9}, {-1}, dtype::Float32()),
            TensorLayout({9}, {1}, dtype::Float32()));

    Checker<Relayout> checker(handle_atlas());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(ATLAS, RELAYOUT_DST_NEG_STRIDE) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    args.emplace_back(
            TensorLayout({9}, {1}, dtype::Float32()),
            TensorLayout({9}, {-1}, dtype::Float32()));

    Checker<Relayout> checker(handle_atlas());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

// vim: syntax=cpp.doxygen
