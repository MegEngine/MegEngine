// generated by gen_elemwise_multi_type_kern_impls.py
#define KERN_IMPL_MODE(cb) MEGDNN_ELEMWISE_MODE_ENABLE(ADD, cb)
#define KERN_IMPL_ARITY    2
#define KERN_IMPL_STYPE    dt_qint32
#define KERN_IMPL_DTYPE    dt_qint8
#include "../kern_impl.inl"
