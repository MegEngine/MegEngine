// generated by gen_cond_take_kern_impls.py
#include "../kern.inl"

#if !MEGDNN_DISABLE_FLOAT16
namespace megdnn {
namespace cuda {
namespace cond_take {

inst_genidx(::megdnn::dtype::Float16)
#undef inst_genidx

        inst_copy(::megdnn::dtype::Float16)
#undef inst_copy
#undef inst_copy_

}  // namespace cond_take
}  // namespace cuda
}  // namespace megdnn
#endif
