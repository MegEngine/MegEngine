#pragma once

#include "megbrain/serialization/internal/dtype_generated.h"
#include "megdnn/dtype.h"

namespace mgb {
namespace serialization {
namespace fbs {
namespace intl {

megdnn::DTypeEnum convert_dtype_to_megdnn(fbs::DTypeEnum fb);
fbs::DTypeEnum convert_dtype_to_fbs(megdnn::DTypeEnum enumv);

megdnn::DType load_dtype(const fbs::DType* dtype);
flatbuffers::Offset<fbs::DType> build_dtype(
        flatbuffers::FlatBufferBuilder& builder, megdnn::DType dtype);

}  // namespace intl
}  // namespace fbs
}  // namespace serialization
}  // namespace mgb
