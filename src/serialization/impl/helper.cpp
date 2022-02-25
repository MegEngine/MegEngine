#include "megbrain/serialization/helper.h"
#include "megbrain/utils/metahelper.h"

using namespace mgb;
using namespace serialization;

void serialization::serialize_dtype(
        DType dtype, megdnn::thin_function<void(const void*, size_t)> write_fn) {
    DTypeEnum enumv = dtype.enumv();
    write_fn(&enumv, sizeof(enumv));
    switch (dtype.enumv()) {
#define cb(_dt)                                                                       \
    case DTypeEnum::_dt:                                                              \
        write_fn(&dtype.param<dtype::_dt>(), sizeof(megdnn::DTypeParam<dtype::_dt>)); \
        break;
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb);
#undef cb
        default:;
    }
}

DType serialization::deserialize_dtype(
        megdnn::thin_function<void(void*, size_t)> read_fn) {
    DTypeEnum enumv;
    read_fn(&enumv, sizeof(enumv));
    switch (enumv) {
#define cb(_dt)          \
    case DTypeEnum::_dt: \
        return DType::from_enum(enumv);
        MEGDNN_FOREACH_DTYPE_NAME(cb)
#undef cb
#define cb(_dt)                               \
    case DTypeEnum::_dt: {                    \
        megdnn::DTypeParam<dtype::_dt> param; \
        read_fn(&param, sizeof(param));       \
        return dtype::_dt{param};             \
    }
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
    }
    mgb_assert(
            false, "unexpected serialized dtype: invalid enumv %d",
            static_cast<uint32_t>(enumv));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
