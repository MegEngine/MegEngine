/**
 * \file dnn/src/atlas/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megcore_atlas.h"
#include "src/common/handle_impl.h"
#include "src/atlas/handle.h"
#include "src/atlas/checksum/opr_impl.h"

#include <acl/acl.h>

namespace megdnn {
namespace atlas {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::ATLAS) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);

    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    m_device_id = dev_id;
    megcore::getAtlasContext(comp_handle, &m_megcore_context);
}

HandleImpl::~HandleImpl() noexcept = default;

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw("unsupported atlas opr");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
    //! because memcpyasync api requires that the memory is 128bytes alignment
    return 64;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
