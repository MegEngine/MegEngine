#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

#include <map>
#include "cnnl.h"
#include "cnrt.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/type_cvt/opr_impl.h"

namespace megdnn {
namespace cambricon {

cnnlCastDataType_t get_cnnl_cast_type(DTypeEnum from, DTypeEnum to) {
    static std::map<std::pair<DTypeEnum, DTypeEnum>, cnnlCastDataType_t>
            cast_dtype_table = {
#if !MEGDNN_DISABLE_FLOAT16
                {{DTypeEnum::Float32, DTypeEnum::Float16}, CNNL_CAST_FLOAT_TO_HALF},
                {{DTypeEnum::Float16, DTypeEnum::Float32}, CNNL_CAST_HALF_TO_FLOAT},
                {{DTypeEnum::Float16, DTypeEnum::Int32}, CNNL_CAST_HALF_TO_INT32},
                {{DTypeEnum::Int32, DTypeEnum::Float16}, CNNL_CAST_INT32_TO_HALF},
                {{DTypeEnum::Float16, DTypeEnum::Int16}, CNNL_CAST_HALF_TO_INT16},
                {{DTypeEnum::Int16, DTypeEnum::Float16}, CNNL_CAST_INT16_TO_HALF},
                {{DTypeEnum::Float16, DTypeEnum::Int8}, CNNL_CAST_HALF_TO_INT8},
                {{DTypeEnum::Int8, DTypeEnum::Float16}, CNNL_CAST_INT8_TO_HALF},
                {{DTypeEnum::Float16, DTypeEnum::Bool}, CNNL_CAST_HALF_TO_BOOL},
                {{DTypeEnum::Bool, DTypeEnum::Float16}, CNNL_CAST_BOOL_TO_HALF},
                {{DTypeEnum::Float16, DTypeEnum::Uint8}, CNNL_CAST_HALF_TO_UINT8},
                {{DTypeEnum::Uint8, DTypeEnum::Float16}, CNNL_CAST_UINT8_TO_HALF},
#endif
                {{DTypeEnum::Float32, DTypeEnum::Int32}, CNNL_CAST_FLOAT_TO_INT32},
                {{DTypeEnum::Float32, DTypeEnum::Int16}, CNNL_CAST_FLOAT_TO_INT16},
                {{DTypeEnum::Float32, DTypeEnum::Int8}, CNNL_CAST_FLOAT_TO_INT8},
                {{DTypeEnum::Float32, DTypeEnum::Uint8}, CNNL_CAST_FLOAT_TO_UINT8},
                {{DTypeEnum::Float32, DTypeEnum::Bool}, CNNL_CAST_FLOAT_TO_BOOL},
                {{DTypeEnum::Int32, DTypeEnum::Float32}, CNNL_CAST_INT32_TO_FLOAT},
                {{DTypeEnum::Int32, DTypeEnum::Int16}, CNNL_CAST_INT32_TO_INT16},
                {{DTypeEnum::Int32, DTypeEnum::Int8}, CNNL_CAST_INT32_TO_INT8},
                {{DTypeEnum::Int32, DTypeEnum::Bool}, CNNL_CAST_INT32_TO_BOOL},
                {{DTypeEnum::Int8, DTypeEnum::Float32}, CNNL_CAST_INT8_TO_FLOAT},
                {{DTypeEnum::Int8, DTypeEnum::Int32}, CNNL_CAST_INT8_TO_INT32},
                {{DTypeEnum::Int8, DTypeEnum::Int16}, CNNL_CAST_INT8_TO_INT16},
                {{DTypeEnum::Uint8, DTypeEnum::Float32}, CNNL_CAST_UINT8_TO_FLOAT},
                {{DTypeEnum::Uint8, DTypeEnum::Int32}, CNNL_CAST_UINT8_TO_INT32},
                {{DTypeEnum::Bool, DTypeEnum::Float32}, CNNL_CAST_BOOL_TO_FLOAT},
                {{DTypeEnum::Bool, DTypeEnum::Int32}, CNNL_CAST_BOOL_TO_INT32},
                {{DTypeEnum::Int16, DTypeEnum::Float32}, CNNL_CAST_INT16_TO_FLOAT},
                {{DTypeEnum::Int16, DTypeEnum::Int32}, CNNL_CAST_INT16_TO_INT32},
            };
    auto it = cast_dtype_table.find(std::make_pair(from, to));
    megdnn_assert(
            it != cast_dtype_table.end(),
            "cambricon cnnl does not support cast tensor from DTypeEnum {%d} to {%d}",
            static_cast<int>(from), static_cast<int>(to));
    return it->second;
}

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    check_exec(src.layout, dst.layout);
    auto from = src.layout.dtype.enumv();
    auto to = dst.layout.dtype.enumv();
    auto handle = cnnl_handle(this->handle());
    CnnlTensorDescriptor from_desc, to_desc;
    from_desc.set(src.layout, convert_to_cnnl_datatype(from));
    to_desc.set(dst.layout, convert_to_cnnl_datatype(to));
    if (from == to) {
        cnnl_check(cnnlCopy(
                handle, from_desc.desc(), src.raw_ptr(), to_desc.desc(),
                dst.raw_ptr()));
    } else {
        auto cnnl_cast_type_ = get_cnnl_cast_type(from, to);
        cnnl_check(cnnlCastDataType(
                handle, from_desc.desc(), src.raw_ptr(), cnnl_cast_type_,
                to_desc.desc(), dst.raw_ptr()));
    }
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen