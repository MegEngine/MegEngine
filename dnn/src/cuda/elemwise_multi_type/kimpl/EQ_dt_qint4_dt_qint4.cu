// generated by gen_elemwise_multi_type_kern_impls.py
#define KERN_IMPL_MODE(cb) MEGDNN_ELEMWISE_MODE_ENABLE(EQ, cb)
#define KERN_IMPL_ARITY    2
#define KERN_IMPL_STYPE    dt_qint4
#define KERN_IMPL_DTYPE    dt_qint4
#include "../kern_impl_q4.inl"
