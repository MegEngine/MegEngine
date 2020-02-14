/**
 * \file src/serialization/impl/flatbuffer_converter.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#ifdef _SRC_SERIALIZATION_IMPL_FLATBUFFER_CONVERTER
#error "flatbuffer_converter.inl should not be included more than once"
#endif

#define _SRC_SERIALIZATION_IMPL_FLATBUFFER_CONVERTER

#include "megbrain/serialization/internal/flatbuffers_helper.h"

namespace mgb {
namespace serialization {
namespace fbs {

template <typename T>
struct ParamConverter;

}  // namespace fbs
}  // namespace serialization
}  // namespace mgb

#include "megbrain/serialization/internal/schema_generated.h"

#include "megdnn/opr_param_defs.h"
#include "opr_param_defs_converter.inl"

#include "megbrain/opr/param_defs.h"
#include "mgb_opr_param_defs_converter.inl"
