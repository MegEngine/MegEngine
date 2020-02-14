/**
 * \file src/serialization/impl/opr_load_dump.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/serialization/opr_load_dump.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/serialization/file.h"
#include "megbrain/serialization/helper.h"

using namespace mgb;
using namespace serialization;

MGB_TYPEINFO_OBJ_IMPL(OprLoadContext);

OprLoader OprLoadContext::make_opr_loader(const std::string &id) {
    auto &&maker = config().opr_loader_maker;
    mgb_throw_if(!maker, SerializationError,
            "opr_loader_maker not set in LoadConfig; but opr loader with "
            "id %s is needed", id.c_str());
    return maker(id);
}

template <>
void OprDumpContextRawPOD::write_param(const DType& param) {
    if (m_check_param_tag) {
        uint32_t tag = megdnn::param::FakeSerializedDType::TAG;
        write_raw(&tag, sizeof(tag));
    }
    serialization::serialize_dtype(param, [this](const void* data, size_t len) {
        write_raw(data, len);
    });
}

template <>
DType OprLoadContextRawPOD::read_param() {
    if (m_check_param_tag) {
        uint32_t tag;
        read_raw(&tag, sizeof(tag));
        mgb_assert(tag == megdnn::param::FakeSerializedDType::TAG);
    }
    return serialization::deserialize_dtype(
            [this](void* data, size_t len) { read_raw(data, len); });
}

std::string OprLoadContextRawPOD::load_buf_with_len() {
    std::string ret;
    uint32_t size;
    read_raw(&size, sizeof(size));
    ret.resize(size);
    read_raw(&ret[0], size);
    return ret;
}

SharedBuffer OprLoadContextRawPOD::load_shared_buf_with_len() {
    uint32_t size;
    read_raw(&size, sizeof(size));
    return load_shared_buf(size);
}

void GraphDumpConfig::default_tensor_value_dumper(
        OutputFile &fout, const cg::OperatorNodeBase &/*opr*/,
        const HostTensorND &tensor) {
    auto size = tensor.layout().span().high_byte;
    fout.write(tensor.raw_ptr(), size);
}

void GraphLoadConfig::default_tensor_value_loader(
        void *ptr, const TensorLayout &layout, InputFile &fin) {
    auto sz = layout.span().high_byte;
    if (ptr) {
        fin.read(ptr, sz);
    } else {
        fin.skip(sz);
    }
}

SharedBuffer OprLoadContextRawPOD::load_shared_buf(size_t size) {
    std::shared_ptr<uint8_t> shptr{new uint8_t[size],
                                   [](uint8_t* p) { delete[] p; }};
    read_raw(shptr.get(), size);
    return {std::move(shptr), size};
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
