// generated by gen_elemwise_multi_type_kern_impls_bool.py
#define KERN_IMPL_MODE(cb) MEGDNN_ELEMWISE_MODE_ENABLE(LEQ, cb)
#define KERN_IMPL_ARITY    2
#define KERN_IMPL_STYPE    dt_int8
#define KERN_IMPL_DTYPE    dt_bool
#include "../kern_impl_bool.inl"
