#include "src/atlas/atlas_wrapper.h"

namespace megdnn::atlas {

aclDataType as_acl_dtype(DTypeEnum dtype) {
#define cb(_acl_dt, _dnn_dt)   \
    case DTypeEnum::_dnn_dt: { \
        return _acl_dt;        \
        break;                 \
    }
    switch (dtype) {
        FOR_ACL_DNN_DTYPE_MAP(cb)
        default:
            megdnn_throw("unsupported dtype");
    }
#undef cb
}

aclDataType as_acl_dtype(DType dtype) {
    return as_acl_dtype(dtype.enumv());
}

DType as_dnn_dtype(aclDataType dt) {
#define cb(_acl_dt, _dnn_dt)     \
    case _acl_dt: {              \
        return dtype::_dnn_dt(); \
        break;                   \
    }
    switch (dt) {
        FOR_ACL_DNN_DTYPE_MAP(cb)
        default:
            megdnn_throw("unsupported dtype");
    }
#undef cb
}

}  // namespace megdnn::atlas
