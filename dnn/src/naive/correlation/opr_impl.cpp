#include "src/naive/correlation/opr_impl.h"
#include <algorithm>
#include "src/common/utils.h"
#include "src/naive/handle.h"
#define ROUND_OFF 50000
using namespace megdnn;
using namespace naive;
using namespace std;
namespace {

using Param = megdnn::Correlation::Param;

template <typename T>
void forward(
        _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
        const Param& param) {
    // data1 treat as no-padding tensor
    int total_nr_elems = dst.layout.total_nr_elems();

    int stride1 = param.stride1, stride2 = param.stride2;
    int kernel_size = param.kernel_size;
    int kernel_radius = (kernel_size - 1) / 2;
    int max_displacement = param.max_displacement;
    int pad_size = param.pad_size;

    int tchannels = dst.layout[1];
    int theight = dst.layout[2], twidth = dst.layout[3];
    int bchannels = data1.layout[1];
    int bheight = data1.layout[2], bwidth = data1.layout[3];

    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

    for (int idx = 0; idx < total_nr_elems; ++idx) {
        int x = idx % twidth;
        int y = (idx / twidth) % theight;
        int c = (idx / twidth / theight) % tchannels;
        int n = idx / twidth / theight / tchannels;

        // get src center position in image1
        int x1 = x * stride1 + kernel_radius + max_displacement - pad_size;
        int y1 = y * stride1 + kernel_radius + max_displacement - pad_size;

        // get offset of center in image2
        int s2o = (c % neighborhood_grid_width - neighborhood_grid_radius) * stride2;
        int s2p = (c / neighborhood_grid_width - neighborhood_grid_radius) * stride2;

        int x2 = x1 + s2o;
        int y2 = y1 + s2p;

        // compute kernel correlation
        float sum = 0.;
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int in_x1 = x1 + i;
                int in_y1 = y1 + j;
                int in_x2 = x2 + i;
                int in_y2 = y2 + j;

                for (int channel = 0; channel < bchannels; channel++) {
                    float tmp1 = 0.;
                    float tmp2 = 0.;
                    if (in_x1 >= 0 && in_x1 < bwidth && in_y1 >= 0 && in_y1 < bheight) {
                        int idx1 =
                                ((n * bchannels + channel) * bheight + in_y1) * bwidth +
                                in_x1;
                        tmp1 = data1.ptr<T>()[idx1];
                    }

                    if (in_x2 >= 0 && in_x2 < bwidth && in_y2 >= 0 && in_y2 < bheight) {
                        int idx2 =
                                ((n * bchannels + channel) * bheight + in_y2) * bwidth +
                                in_x2;
                        tmp2 = data2.ptr<T>()[idx2];
                    }

                    if (param.is_multiply) {
                        sum += tmp1 * tmp2;
                    } else {
                        sum += fabsf(tmp1 - tmp2);
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        dst.ptr<T>()[idx] = sum / sumelems;
    }
}

template <typename T>
void backward_data1(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad1, const Param& param) {
    // data1 treat as no-padding tensor
    // int total_nr_elems = diff.layout.total_nr_elems();
    int total_nr_elems = grad1.layout.total_nr_elems();

    int stride1 = param.stride1, stride2 = param.stride2;
    int kernel_size = param.kernel_size;
    int kernel_radius = (kernel_size - 1) / 2;
    int max_displacement = param.max_displacement;
    int pad_size = param.pad_size;

    int tchannels = diff.layout[1];
    int theight = diff.layout[2], twidth = diff.layout[3];
    int bchannels = grad1.layout[1];
    int bheight = grad1.layout[2], bwidth = grad1.layout[3];

    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

    for (int idx = 0; idx < total_nr_elems; ++idx) {
        // idx for grad1

        int x = idx % bwidth;
        int y = (idx / bwidth) % bheight;
        int c = (idx / bwidth / bheight) % bchannels;
        int n = idx / bwidth / bheight / bchannels;

        float tmp1 = data1.ptr<T>()[idx];
        // Get X,Y ranges and clamp
        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers We use a large offset, for the inner part not to
        // become negative.
        const int round_off = ROUND_OFF;
        const int round_off_s1 = stride1 * round_off;

        // we show cal the x_min,y_min,x_max,y_max of diff for grad1(x,y)
        // for diff_x_min, diff_y_min, x,y at the position of right-down
        // ceil (l - 2*kernel_radius - max_displacement + pad_size) / stride1
        int xmin = (x + pad_size - 2 * kernel_radius - max_displacement + round_off_s1 -
                    1) / stride1 +
                   1 - round_off;
        int ymin = (y + pad_size - 2 * kernel_radius - max_displacement + round_off_s1 -
                    1) / stride1 +
                   1 - round_off;
        // floor (l - max_displacement + pad_size) / stride1
        int xmax =
                (x + pad_size - max_displacement + round_off_s1) / stride1 - round_off;
        int ymax =
                (y + pad_size - max_displacement + round_off_s1) / stride1 - round_off;

        float sum = 0.;
        if (xmax >= 0 && ymax >= 0 && (xmin <= twidth - 1) && (ymin <= theight - 1)) {
            xmin = max(0, xmin);
            xmax = min(twidth - 1, xmax);

            ymin = max(0, ymin);
            ymax = min(theight - 1, ymax);

            for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius;
                 p++) {
                for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius;
                     o++) {
                    // Get bottom1 data:
                    int s2o = stride2 * o;
                    int s2p = stride2 * p;
                    int x2 = x + s2p, y2 = y + s2o;

                    int idx2 = ((n * bchannels + c) * bheight + y2) * bwidth + x2;
                    float tmp2 = 0.;

                    if (x2 >= 0 && x2 < bwidth && y2 >= 0 && y2 < bheight) {
                        tmp2 = data2.ptr<T>()[idx2];
                    }

                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width +
                             (o + neighborhood_grid_radius);
                    int diff_channels_offset = (n * tchannels + op);

                    for (int diff_y = ymin; diff_y <= ymax; diff_y++) {
                        for (int diff_x = xmin; diff_x <= xmax; diff_x++) {
                            int idxtopdiff =
                                    (diff_channels_offset * theight + diff_y) * twidth +
                                    diff_x;

                            if (param.is_multiply) {
                                sum += diff.ptr<T>()[idxtopdiff] * tmp2;
                            } else {
                                T sign = (tmp1 > tmp2) ? T(1.) : T(-1.);
                                sum += diff.ptr<T>()[idxtopdiff] * sign;
                            }
                        }
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        grad1.ptr<T>()[idx] = sum / sumelems;
    }
}

template <typename T>
void backward_data2(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad2, const Param& param) {
    // data1 treat as no-padding tensor
    int total_nr_elems = grad2.layout.total_nr_elems();

    int stride1 = param.stride1, stride2 = param.stride2;
    int kernel_size = param.kernel_size;
    int kernel_radius = (kernel_size - 1) / 2;
    int max_displacement = param.max_displacement;
    int pad_size = param.pad_size;

    int tchannels = diff.layout[1];
    int theight = diff.layout[2], twidth = diff.layout[3];
    int bchannels = grad2.layout[1];
    int bheight = grad2.layout[2], bwidth = grad2.layout[3];

    int neighborhood_grid_radius = max_displacement / stride2;
    int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

    for (int idx = 0; idx < total_nr_elems; ++idx) {
        int x = idx % bwidth;
        int y = (idx / bwidth) % bheight;
        int c = (idx / bwidth / bheight) % bchannels;
        int n = idx / bwidth / bheight / bchannels;

        T tmp2 = data2.ptr<T>()[idx];

        T sum = T(0.f);

        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius; p++) {
            for (int o = -neighborhood_grid_radius; o <= neighborhood_grid_radius;
                 o++) {
                int s2o = o * stride2;
                int s2p = p * stride2;

                int x1 = x - s2o;
                int y1 = y - s2p;

                const int round_off = ROUND_OFF;
                const int round_off_s1 = stride1 * round_off;

                int xmin = (x1 + pad_size - 2 * kernel_radius - max_displacement +
                            round_off_s1 - 1) /
                                   stride1 +
                           1 - round_off;
                int ymin = (y1 + pad_size - 2 * kernel_radius - max_displacement +
                            round_off_s1 - 1) /
                                   stride1 +
                           1 - round_off;
                int xmax = (x1 + pad_size - max_displacement + round_off_s1) / stride1 -
                           round_off;
                int ymax = (y1 + pad_size - max_displacement + round_off_s1) / stride1 -
                           round_off;

                if (xmax >= 0 && ymax >= 0 && (xmin <= twidth - 1) &&
                    (ymin <= theight - 1)) {
                    xmin = max(0, xmin);
                    xmax = min(twidth - 1, xmax);

                    ymin = max(0, ymin);
                    ymax = min(theight - 1, ymax);

                    int idx1 = ((n * bchannels + c) * bheight + y1) * bwidth + x1;
                    T tmp1 = T(0.f);
                    if (x1 >= 0 && x1 < bwidth && y1 >= 0 && y1 < bheight) {
                        tmp1 = data1.ptr<T>()[idx1];
                    }

                    int op = (p + neighborhood_grid_radius) * neighborhood_grid_width +
                             (o + neighborhood_grid_radius);
                    int diff_channels_offset = (n * tchannels + op);
                    for (int diff_y = ymin; diff_y <= ymax; diff_y++) {
                        for (int diff_x = xmin; diff_x <= xmax; diff_x++) {
                            int idxtopdiff =
                                    (diff_channels_offset * theight + diff_y) * twidth +
                                    diff_x;

                            if (param.is_multiply) {
                                sum += diff.ptr<T>()[idxtopdiff] * tmp1;
                            } else {
                                T sign = (tmp1 >= tmp2) ? T(-1.f) : T(1.f);
                                sum += diff.ptr<T>()[idxtopdiff] * sign;
                            }
                        }
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        grad2.ptr<T>()[idx] = sum / sumelems;
    }
}

}  // namespace

namespace megdnn {
namespace naive {

void CorrelationForwardImpl::exec(
        _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(data1.layout, data2.layout, dst.layout, workspace.size);
#define cb(DType)                                                                \
    if (data1.layout.dtype == DType()) {                                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<typename DTypeTrait<DType>::ctype>( \
                data1, data2, dst, param()));                                    \
        return;                                                                  \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
#else
    __builtin_trap();
#endif
}

void CorrelationBackwardData1Impl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad1, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(diff.layout, data1.layout, data2.layout, grad1.layout, workspace.size);
#define cb(DType)                                                  \
    if (diff.layout.dtype == DType()) {                            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                              \
                backward_data1<typename DTypeTrait<DType>::ctype>( \
                        diff, data1, data2, grad1, param()));      \
        return;                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
#else
    __builtin_trap();
#endif
}

void CorrelationBackwardData2Impl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad2, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(diff.layout, data1.layout, data2.layout, grad2.layout, workspace.size);
#define cb(DType)                                                  \
    if (diff.layout.dtype == DType()) {                            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                              \
                backward_data2<typename DTypeTrait<DType>::ctype>( \
                        diff, data1, data2, grad2, param()));      \
        return;                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
#else
    __builtin_trap();
#endif
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
