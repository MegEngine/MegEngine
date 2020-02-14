/**
 * \file src/serialization/include/megbrain/serialization/internal/flatbuffers_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
