#include "megdnn/oprs.h"
#include "src/common/relayout_helper.h"
#include "src/common/utils.h"

#include <algorithm>

using namespace megdnn;
using namespace megdnn::relayout;

namespace {

//! whether current shape is [b][n][m][c] and is a transpose of contig
//! [b][m][n][c]
bool is_transpose_single(
        const TensorLayout& layout, TransposeParam& p, bool allow_no_contig) {
    /*
     * assuming contig layout is:
     *  shape: b, m, n, c
     *  stride: mnc, nc, c, 1
     * assuming non-contig layout is:
     *  shape: b, m, n, c
     *  stride: m*stride_m*c, stride_m*c, c, 1
     *
     * then given layout should be:
     *  shape: b, n, m, c
     *  stride: mnc, c, nc, 1
     *  non-contig stride: m*stride_m*c, c, stride_m*c, 1
     *
     * if c == 1:
     *  shape: b, n, m
     *  stride: mn, 1, n
     *  non-contig stride: m*stride_m, 1, stride_m
     *
     * if b == 1:
     *  shape: n, m, c
     *  stride: c, nc, 1
     *  non-contig stride: c, stride_m*c, 1
     *
     * if b == 1 && c == 1:
     *  shape: n, m
     *  stride: 1, n(stride_m for no-contig)
     */
    p.stride_m = 0;
    auto strd = [&](size_t idx, ptrdiff_t v) { return layout.stride[idx] == v; };
    if (layout.ndim == 4) {
        p.batch = layout[0];
        p.n = layout[1];
        p.m = layout[2];
        p.c = layout[3];
        if (strd(3, 1) && strd(1, p.c)) {
            auto t = p.c * p.n;
            return strd(2, t) && strd(0, t * p.m);
        }
        return false;
    }
    if (layout.ndim == 3) {
        if (strd(1, 1)) {
            // c == 1
            p.batch = layout[0];
            p.n = layout[1];
            p.m = layout[2];
            p.c = 1;

            if (strd(2, p.n) && strd(0, p.m * p.n)) {
                return true;
            } else if (
                    allow_no_contig && (size_t)(layout.stride[2]) >= p.n &&
                    strd(0, p.m * (size_t)(layout.stride[2])) && strd(1, 1)) {
                p.stride_m = layout.stride[2];
                return true;
            }
            return false;
        }
        if (strd(2, 1)) {
            // b == 1
            p.batch = 1;
            p.n = layout[0];
            p.m = layout[1];
            p.c = layout[2];
            return strd(0, p.c) && strd(1, p.n * p.c);
        }
        return false;
    }
    if (layout.ndim == 2) {
        p.batch = 1;
        p.n = layout.shape[0];
        p.m = layout.shape[1];
        p.c = 1;
        if (strd(0, 1) && strd(1, p.n)) {
            return true;
        } else if (
                strd(0, 1) && layout.stride[1] > 0 &&
                (size_t)(layout.stride[1]) >= p.n && allow_no_contig) {
            //! stride_m used in no-contig mode, stride_m >= p.n
            p.stride_m = layout.stride[1];
            return true;
        }
    }
    return false;
}

}  // anonymous namespace

void RelayoutForward::check_layout_and_canonize(TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(dst.is_non_overlapping_strong());
    src = src.collapse_contiguous();
    dst = dst.collapse_contiguous();
    megdnn_assert(
            src.dtype == dst.dtype && src.total_nr_elems() == dst.total_nr_elems(),
            "check %s == %s and %zu == %zu", src.dtype.name(), dst.dtype.name(),
            src.total_nr_elems(), dst.total_nr_elems());
}

bool relayout::is_transpose(
        const TensorLayout& src, const TensorLayout& dst, TransposeParam& p,
        bool allow_non_contig) {
    if (is_contig(dst) && is_transpose_single(src, p, allow_non_contig)) {
        // if the original intention is to transpose (m, n) to (n, m),
        // then we should use (n, m) as the contig dst and use a corrsponding
        // non-contig src with the same (n, m) shape (remember relayout is
        // defined on element correspondence on the logical view)
        return true;
    }
    if (is_contig(src) && is_transpose_single(dst, p, allow_non_contig)) {
        std::swap(p.m, p.n);
        return true;
    }
    return false;
}

// vim: syntax=cpp.doxygen
