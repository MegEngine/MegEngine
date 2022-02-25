#include "./opr_impl.h"
#include "megdnn/tensor_iter.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

template <typename T>
void PowCImpl::do_exec_ct(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
        const int* exp_i) {
    if (exp_i) {
        auto kern = [src, dst, iv = *exp_i]() {
            auto src_iter = tensor_iter_valonly<T>(src).begin();
            auto dst_iter = tensor_iter_valonly<T>(dst).begin();
            T ivt = static_cast<T>(iv);
            for (size_t i = 0, it = src.layout.total_nr_elems(); i < it; ++i) {
                T sv = *src_iter;
                T dv = static_cast<T>(std::pow(std::abs(sv), ivt));
                if (iv && (iv & 1) && sv < 0) {
                    dv = -dv;
                }
                *dst_iter = dv;
                ++dst_iter;
                ++src_iter;
            }
        };
        static_cast<HandleImpl*>(this->handle())->dispatch_kern(kern);
    } else {
        auto kern = [src, dst, fv = *exp_f]() {
            auto src_iter = tensor_iter_valonly<T>(src).begin();
            auto dst_iter = tensor_iter_valonly<T>(dst).begin();
            T fvt = static_cast<T>(fv);
            for (size_t i = 0, it = src.layout.total_nr_elems(); i < it; ++i) {
                *dst_iter = static_cast<T>(std::pow(*src_iter, fvt));
                ++dst_iter;
                ++src_iter;
            }
        };
        static_cast<HandleImpl*>(this->handle())->dispatch_kern(kern);
    }
}

void PowCImpl::do_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
        const int* exp_i) {
    switch (src.layout.dtype.enumv()) {
#define cb(dt)                  \
    case DTypeTrait<dt>::enumv: \
        return do_exec_ct<DTypeTrait<dt>::ctype>(src, dst, exp_f, exp_i);
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            megdnn_throw("unsupported dtype for PowC");
    }
}

// vim: syntax=cpp.doxygen
