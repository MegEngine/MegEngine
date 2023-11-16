#include "megdnn/oprs.h"

#include "test/common/checker.h"
#include "test/common/tensor.h"

#include "test/cambricon/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CAMBRICON, RELAYOUT_TRANSPOSE) {
    Checker<RelayoutForward> checker(handle_cambricon());
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    auto args_add = [&](megdnn::DType dtype) {
        args.emplace_back(
                TensorLayout({3, 2}, {2, 1}, dtype),
                TensorLayout({3, 2}, {1, 3}, dtype));
        args.emplace_back(
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype),
                TensorLayout({4, 3, 2}, {2, 8, 1}, dtype));
        args.emplace_back(
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype),
                TensorLayout({4, 3, 2}, {1, 4, 12}, dtype));
        args.emplace_back(
                TensorLayout({8, 4, 3, 2}, {24, 6, 2, 1}, dtype),
                TensorLayout({8, 4, 3, 2}, {24, 2, 8, 1}, dtype));
        args.emplace_back(
                TensorLayout({8, 4, 3, 2}, {24, 6, 2, 1}, dtype),
                TensorLayout({8, 4, 3, 2}, {1, 8, 32, 96}, dtype));
        args.emplace_back(
                TensorLayout({3, 2}, {1, 3}, dtype),
                TensorLayout({3, 2}, {2, 1}, dtype));
        args.emplace_back(
                TensorLayout({4, 3, 2}, {6, 1, 3}, dtype),
                TensorLayout({4, 3, 2}, {6, 2, 1}, dtype));
        args.emplace_back(
                TensorLayout({8, 4, 3, 2}, {24, 3, 1, 12}, dtype),
                TensorLayout({8, 4, 3, 2}, {24, 6, 2, 1}, dtype));
        args.emplace_back(
                TensorLayout({8, 4, 3, 2}, {24, 1, 8, 4}, dtype),
                TensorLayout({8, 4, 3, 2}, {24, 6, 2, 1}, dtype));
    };
    args_add(dtype::Uint8());
    args_add(dtype::Int8());
    args_add(dtype::Uint16());
    args_add(dtype::Int16());
    args_add(dtype::Int32());
    args_add(dtype::Bool());
    args_add(dtype::Float16());
    args_add(dtype::Float32());

    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(CAMBRICON, RELAYOUT) {
    Checker<RelayoutForward> checker(handle_cambricon());
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };

    std::vector<Arg> args;
    {
        auto args_add = [&](megdnn::DType dtype) {
            args.emplace_back(
                    TensorLayout({4, 3, 2}, {2, 8, 1}, dtype),
                    TensorLayout({4, 3, 2}, {6, 2, 1}, dtype));
            args.emplace_back(
                    TensorLayout({4, 3, 2}, {6, 2, 1}, dtype),
                    TensorLayout({4, 3, 2}, {2, 8, 1}, dtype));
            args.emplace_back(
                    TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype),
                    TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype));
            args.emplace_back(
                    TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype),
                    TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype));
        };
        args_add(dtype::Uint8());
        args_add(dtype::Int8());
        args_add(dtype::Uint16());
        args_add(dtype::Int16());
        args_add(dtype::Int32());
        args_add(dtype::Bool());
        args_add(dtype::Float16());
        args_add(dtype::Float32());
    }
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float32()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Float32()));
    args.emplace_back(
            TensorLayout({16, 128, 128}, {49152, 384, 3}, dtype::Float32()),
            TensorLayout({16, 128, 128}, {16384, 128, 1}, dtype::Float32()));
    {
        // broadcast
        for (size_t n : {128, 256, 1024, 2048}) {
            args.emplace_back(
                    TensorLayout({n}, {0}, dtype::Float32()),
                    TensorLayout({n}, {1}, dtype::Float32()));
        }
    }
    {
        // 1d
        size_t n = 10000;
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Float32()),
                TensorLayout({n}, {1}, dtype::Float32()));
        args.emplace_back(
                TensorLayout({n}, {1}, dtype::Float32()),
                TensorLayout({n}, {2}, dtype::Float32()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Float32()),
                TensorLayout({n}, {1}, dtype::Float32()));
        args.emplace_back(
                TensorLayout({n}, {2}, dtype::Float32()),
                TensorLayout({n}, {2}, dtype::Float32()));
    }
    {
        // 2d
        auto args_add = [&](megdnn::DType dtype) {
            size_t m = 20, n = 30, k = 40;
            ptrdiff_t k2 = k * 2;
            args.emplace_back(
                    TensorLayout({m, n}, {k2, 2}, dtype),
                    TensorLayout({m, n}, {k2 + 1, 2}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {2, k2}, dtype),
                    TensorLayout({m, n}, {2, k2 + 1}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {2, k2}, dtype),
                    TensorLayout({m, n}, {k2 + 1, 2}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {k2, 2}, dtype),
                    TensorLayout({m, n}, {2, k2 + 1}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {k2, 1}, dtype),
                    TensorLayout({m, n}, {k2 + 1, 1}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {1, k2}, dtype),
                    TensorLayout({m, n}, {1, k2 + 1}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {1, k2}, dtype),
                    TensorLayout({m, n}, {k2 + 1, 1}, dtype));
            args.emplace_back(
                    TensorLayout({m, n}, {k2, 1}, dtype),
                    TensorLayout({m, n}, {1, k2 + 1}, dtype));
        };
        args_add(dtype::Int8());
        args_add(dtype::Float16());
        args_add(dtype::Float32());
        // TODO: test with 8 bytes.
    }
    {
        // 3d
        size_t m = 20, n = 30, k = 40;
        ptrdiff_t k2 = k;
        args.emplace_back(
                TensorLayout({m, n, k}, {k2 * k2 * 4, k2 * 3, 2}, dtype::Float32()),
                TensorLayout(
                        {m, n, k}, {2 * k2 * k2 * k2 * 4, k2 * 3, 2},
                        dtype::Float32()));
    }
    {
        // simplify_layout
        // 234..56
        // 2..3456
        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Float32()),
                TensorLayout(
                        {2, 3, 4, 5, 6}, {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                        dtype::Float32()));
    }
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(CAMBRICON, RELAYOUT_TEST) {
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

    Checker<Relayout> checker(handle_cambricon());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
