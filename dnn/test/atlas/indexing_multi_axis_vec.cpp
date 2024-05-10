#include "test/atlas/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/index.h"
#include "test/common/indexing_multi_axis_vec.h"

#include <random>

using namespace megdnn;
using namespace test;

namespace {

class OrderedRNG final : public RNG {
public:
    void gen(const TensorND& tensor) override {
        auto span = tensor.layout.span();
        if (tensor.layout.dtype == dtype::Float32()) {
            auto ptr = tensor.ptr<float>() + span.low_elem;
            for (size_t i = 0, it = span.dist_elem(); i < it; ++i) {
                ptr[i] = i;
            }
        } else if (tensor.layout.dtype == dtype::Float16()) {
            auto ptr = tensor.ptr<dt_float16>() + span.low_elem;
            for (size_t i = 0, it = span.dist_elem(); i < it; ++i) {
                ptr[i] = i;
            }
        } else if (tensor.layout.dtype == dtype::Bool()) {
            auto ptr = tensor.ptr<bool>() + span.low_elem;
            for (size_t i = 0, it = span.dist_elem(); i < it; ++i) {
                ptr[i] = i % 3;
            }
        } else {
            auto ptr = tensor.ptr<int>() + span.low_elem;
            for (size_t i = 0, it = span.dist_elem(); i < it; ++i) {
                ptr[i] = i;
            }
        }
    }
};

template <class Opr>
void run_check(Handle* handle) {
    // see OprProxyIndexingMultiAxisVecHelper for more details
    // set_proxy() sets the axes to index on
    // execs() give input, output and index layouts

    Checker<Opr> checker(handle);
    size_t idx_size0, idx_size1, idx_size2;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3}, rng2{idx_size2, 4};
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_dtype(3, dtype::Int32())
            .  // idx1
            set_dtype(4, dtype::Int32())
            .  // idx2
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0)
            .set_rng(3, &rng1)
            .set_rng(4, &rng2);

    idx_size0 = 23;
    checker.set_proxy({{0}})
            .execs({{23}, {100}, {100}})
            .execs({{23, 5}, {100, 5}, {100}});

    idx_size0 = 2;
    idx_size1 = 3;
    checker.set_proxy({{0, 1}})
            .execs({{2, 3}, {10}, {10}, {10}})
            .execs({{2, 3, 5}, {10, 5}, {10}, {10}});

    idx_size0 = 2;
    idx_size1 = 3;
    checker.set_proxy({{0, 1}}).execl(
            {{{2, 3, 5}, dtype::Float32()},
             {{10, 8, 5}, dtype::Float32()},
             {{10, 1}, dtype::Int32()},
             {{10, 8}, dtype::Int32()}});

    idx_size0 = 2;
    idx_size1 = 3;
    checker.set_proxy({{0, 1}}).execl(
            {{{2, 3, 5}, {40, 12, 2}, dtype::Float32()},
             {{10, 8, 5}, dtype::Float32()},
             {{10, 1}, dtype::Int32()},
             {{10, 8}, dtype::Int32()}});

    idx_size0 = 2;
    idx_size1 = 5;
    checker.set_proxy({{0, 2}}).execl(
            {{{2, 3, 5}, {40, 12, 2}, dtype::Float32()},
             {{10, 8, 3}, dtype::Float32()},
             {{10, 1}, dtype::Int32()},
             {{10, 8}, dtype::Int32()}});

    checker.set_proxy({{0, 2}}).execl(
            {{{2, 3, 5}, {40, 12, 2}, dtype::Float32()},
             {{10, 3}, dtype::Float32()},
             {{10}, dtype::Int32()},
             {{1}, dtype::Int32()}});

    checker.set_proxy({{0, 2}}).execl(
            {{{2, 3, 5, 6}, {280, 90, 16, 2}, dtype::Float32()},
             {{10, 3, 6}, dtype::Float32()},
             {{10}, dtype::Int32()},
             {{1}, dtype::Int32()}});

    checker.set_proxy({{0, 2}}).execl(
            {{{2, 3, 5}, {40, 12, 2}, dtype::Float32()},
             {{10, 8, 3}, {48, 6, 2}, dtype::Float32()},
             {{10, 1}, dtype::Int32()},
             {{10, 8}, dtype::Int32()}});

    idx_size0 = 4;
    idx_size1 = 6;
    TensorLayout inp_layout{{3, 4, 5, 6}, dtype::Float32()};
    inp_layout.stride[0] *= 8;
    inp_layout.stride[1] *= 2;
    checker.set_proxy({{1, 3}}).execl({
            inp_layout,
            {{7, 3, 5}, dtype::Float32()},
            {{7}, dtype::Int32()},
            {{1}, dtype::Int32()},
    });

    idx_size0 = 3;
    idx_size1 = 5;
    idx_size2 = 7;
    TensorLayout inp_layout_for_ternary_axes{{3, 4, 5, 6, 7}, dtype::Float32()};
    inp_layout.stride[0] *= 8;
    inp_layout.stride[1] *= 2;
    checker.set_proxy({{0, 2, 4}})
            .execl({
                    inp_layout_for_ternary_axes,
                    {{7, 4, 6}, dtype::Float32()},
                    {{7}, dtype::Int32()},
                    {{1}, dtype::Int32()},
                    {{1}, dtype::Int32()},
            });

    idx_size0 = 4;
    idx_size1 = 6;
    TensorLayout value_layout{{7, 3, 5}, dtype::Float32()};
    value_layout.stride[2] *= 2;
    value_layout.stride[1] = 2 * 5;
    value_layout.stride[0] = 2 * 5 * 3;
    TensorLayout idx1_layout = {{7}, dtype::Int32()};
    idx1_layout.stride[0] *= 3;
    TensorLayout idx2_layout = {{1}, dtype::Int32()};
    idx2_layout.stride[0] *= 5;
    checker.set_proxy({{1, 3}}).execl({
            inp_layout,
            value_layout,
            idx1_layout,
            idx2_layout,
    });

    idx_size0 = 4;
    idx_size1 = 6;
    checker.set_proxy({{1, 3}}).execl({
            {{3, 4, 5, 6}, dtype::Float32()},
            {{7, 3, 5}, dtype::Float32()},
            idx1_layout,
            idx2_layout,
    });

    idx_size0 = 4;
    idx_size1 = 6;
    checker.set_proxy({{1, 3}}).execl({
            {{3, 4, 5, 6}, dtype::Float32()},
            value_layout,
            {{7}, dtype::Int32()},
            {{1}, dtype::Int32()},
    });

    idx_size0 = 4;
    idx_size1 = 6;
    checker.set_proxy({{1, 3}}).execl({
            {{3, 4, 5, 6}, dtype::Float32()},
            value_layout,
            idx1_layout,
            idx2_layout,
    });

    idx_size0 = 4;
    checker.set_proxy({{1}}).execs({{1, 4}, {1, 6 * 6}, {6 * 6}});

    idx_size0 = 3;
    checker.set_proxy({{0}}).execs({{3, 4, 5}, {2, 2, 4, 5}, {2, 2}});

    if (std::is_same<Opr, IndexingIncrMultiAxisVec>::value) {
        idx_size0 = 4;
        TensorLayout val_layout{{23}, dtype::Float32()};
        val_layout.stride[0] = 0;
        checker.set_proxy({{0}}).execl(
                {{{4}, dtype::Float32()}, val_layout, {{23}, dtype::Int32()}});
    }
}

template <class Opr>
void run_check_large_ndim(Handle* handle) {
    // see OprProxyIndexingMultiAxisVecHelper for more details
    // set_proxy() sets the axes to index on
    // execs() give input, output and index layouts

    Checker<Opr> checker(handle);
    size_t idx_size0, idx_size1, idx_size2;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3}, rng2{idx_size2, 4};
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_dtype(3, dtype::Int32())
            .  // idx1
            set_dtype(4, dtype::Int32())
            .  // idx2
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0)
            .set_rng(3, &rng1)
            .set_rng(4, &rng2);

    idx_size0 = 2;
    idx_size1 = 2;
    checker.set_proxy({{1, 2}}).execs({{2, 2, 2, 2, 2, 2}, {2, 3, 2, 2, 2}, {3}, {3}});

    idx_size0 = 4;
    idx_size1 = 5;
    checker.set_proxy({{2, 3}}).execs(
            {{2, 3, 4, 5, 6, 1}, {2, 3, 10, 6, 1}, {10}, {10}});

    idx_size0 = 4;
    idx_size1 = 5;
    checker.set_proxy({{2, 3}}).execs(
            {{2, 3, 4, 5, 6, 7}, {2, 3, 10, 6, 7}, {10}, {10}});
}

template <class Opr>
void run_check_index_neg_stride(Handle* handle) {
    Checker<Opr> checker(handle);
    size_t idx_size0;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2};
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    idx_size0 = 20;
    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, dtype::Float32()}, TensorLayout{{9}, dtype::Float32()},
             TensorLayout{TensorShape{9}, {-1}, dtype::Int32()}});

    checker.set_dtype(0, dtype::Float16())
            .  // data
            set_dtype(1, dtype::Float16())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, dtype::Float16()}, TensorLayout{{9}, dtype::Float16()},
             TensorLayout{TensorShape{9}, {-1}, dtype::Int32()}});
}

template <class Opr>
void run_check_src_neg_stride(Handle* handle) {
    Checker<Opr> checker(handle);
    size_t idx_size0;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2};
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    idx_size0 = 20;
    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, {-1}, dtype::Float32()},
             TensorLayout{{9}, dtype::Float32()},
             TensorLayout{TensorShape{9}, dtype::Int32()}});

    checker.set_dtype(0, dtype::Float16())
            .  // data
            set_dtype(1, dtype::Float16())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, {-1}, dtype::Float16()},
             TensorLayout{{9}, dtype::Float16()},
             TensorLayout{TensorShape{9}, dtype::Int32()}});
}

template <class Opr>
void run_check_dst_neg_stride(Handle* handle) {
    Checker<Opr> checker(handle);
    size_t idx_size0;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2};
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    idx_size0 = 20;
    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, dtype::Float32()},
             TensorLayout{{9}, {-1}, dtype::Float32()},
             TensorLayout{TensorShape{9}, dtype::Int32()}});

    checker.set_dtype(0, dtype::Float16())
            .  // data
            set_dtype(1, dtype::Float16())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng_inp)
            .set_rng(1, &rng_inp)
            .set_rng(2, &rng0);

    checker.set_proxy({{0}}).execl(
            {TensorLayout{{20}, dtype::Float16()},
             TensorLayout{{9}, {-1}, dtype::Float16()},
             TensorLayout{TensorShape{9}, dtype::Int32()}});
}

}  // namespace

TEST_F(ATLAS, INDEXING_MULTI_AXIS_VEC) {
    run_check<IndexingMultiAxisVec>(handle_atlas());
    run_check_large_ndim<IndexingMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_MULTI_AXIS_VEC_INDEX_NEG_STRIDE) {
    run_check_index_neg_stride<IndexingMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_MULTI_AXIS_VEC_SRC_NEG_STRIDE) {
    run_check_src_neg_stride<IndexingMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_MULTI_AXIS_VEC_DST_NEG_STRIDE) {
    run_check_dst_neg_stride<IndexingMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_MULTI_AXIS_VEC_ND_INDEX) {
    run_check<IndexingMultiAxisVec>(handle_atlas());
    Checker<IndexingMultiAxisVec> checker(handle_atlas());
    OrderedRNG rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_rng(4, &rng);

    checker.set_proxy({{1, 2, 3}})
            .execs({{5, 5, 6, 7, 3}, {5, 2, 3, 4, 3}, {3, 1}, {2, 1, 1}, {1, 4}});

    size_t idx_size0, idx_size1, idx_size2;
    IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3}, rng2{idx_size2, 4};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng0)
            .set_rng(3, &rng1)
            .set_rng(4, &rng2);

    idx_size0 = 5;
    idx_size1 = 6;
    idx_size2 = 7;
    checker.set_proxy({{1, 2, 3}})
            .execs({{5, 5, 6, 7, 3}, {5, 2, 3, 4, 3}, {3, 1}, {2, 1, 1}, {3, 4}})
            .execs({{5, 5, 6, 7, 3}, {5, 2, 3, 1, 3}, {3, 1}, {2, 1, 1}, {3, 1}});

    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::Int32())
            .set_dtype(4, dtype::Int32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_rng(4, &rng);

    checker.set_proxy({{1, 2, 3}})
            .execs({{5, 5, 6, 7, 3}, {5, 2, 3, 4, 3}, {3, 1}, {2, 1, 1}, {1, 4}});
}

TEST_F(ATLAS, INDEXING_SET_MULTI_AXIS_VEC) {
    Checker<IndexingSetMultiAxisVec> checker(handle_atlas());
    OrderedRNG rng;
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    checker.set_proxy({{1}}).execs({{5, 8, 3}, {5, 2, 3}, {2}});

    checker.set_dtype(0, dtype::Float16())
            .  // data
            set_dtype(1, dtype::Float16())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    checker.set_proxy({{1}}).execs({{5, 8, 3}, {5, 2, 3}, {2}});
}

TEST_F(ATLAS, INDEXING_SET_MULTI_AXIS_VEC_INDEX_NEG_STRIDE) {
    run_check_index_neg_stride<IndexingSetMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_SET_MULTI_AXIS_VEC_SRC_NEG_STRIDE) {
    run_check_src_neg_stride<IndexingSetMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_SET_MULTI_AXIS_VEC_DST_NEG_STRIDE) {
    run_check_dst_neg_stride<IndexingSetMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_INCR_MULTI_AXIS_VEC) {
    run_check<IndexingIncrMultiAxisVec>(handle_atlas());
    run_check_large_ndim<IndexingIncrMultiAxisVec>(handle_atlas());
    Checker<IndexingIncrMultiAxisVec> checker(handle_atlas());
    OrderedRNG rng;
    checker.set_dtype(0, dtype::Float32())
            .  // data
            set_dtype(1, dtype::Float32())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    checker.set_proxy({{1}}).execs({{5, 8, 3}, {5, 2, 3}, {2}});

    checker.set_dtype(0, dtype::Float16())
            .  // data
            set_dtype(1, dtype::Float16())
            .  // value
            set_dtype(2, dtype::Int32())
            .  // idx0
            set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);

    checker.set_proxy({{1}}).execs({{5, 8, 3}, {5, 2, 3}, {2}});
}

TEST_F(ATLAS, INDEXING_INCR_MULTI_AXIS_VEC_INDEX_NEG_STRIDE) {
    run_check_index_neg_stride<IndexingIncrMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_INCR_MULTI_AXIS_VEC_SRC_NEG_STRIDE) {
    run_check_src_neg_stride<IndexingIncrMultiAxisVec>(handle_atlas());
}

TEST_F(ATLAS, INDEXING_INCR_MULTI_AXIS_VEC_DST_NEG_STRIDE) {
    run_check_dst_neg_stride<IndexingIncrMultiAxisVec>(handle_atlas());
}