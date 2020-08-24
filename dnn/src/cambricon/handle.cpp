/**
 * \file dnn/src/cambricon/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"

#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"

#include "src/cambricon/checksum/opr_impl.h"
#include <cnrt.h>

namespace megdnn {
namespace cambricon {

HandleImpl::HandleImpl(megcoreComputingHandle_t comp_handle)
        : HandleImplHelper(comp_handle, HandleType::CAMBRICON) {
    // Get megcore device handle
    megcoreDeviceHandle_t dev_handle;
    megcoreGetDeviceHandle(comp_handle, &dev_handle);
    int dev_id;
    megcoreGetDeviceID(dev_handle, &dev_id);
    unsigned int dev_num;
    cnrt_check(cnrtGetDeviceCount(&dev_num));
    MEGDNN_MARK_USED_VAR(dev_num);
    // check validity of device_id
    megdnn_assert(dev_id >= 0 && static_cast<unsigned int>(dev_id) < dev_num);
    m_device_id = dev_id;
    cnrt_check(cnrtGetDeviceInfo(&m_device_info, dev_id));
    megcore::getCambriconContext(comp_handle, &m_megcore_context);
}

HandleImpl::~HandleImpl() noexcept = default;

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    megdnn_throw("unsupported cambricon opr");
    return nullptr;
}

size_t HandleImpl::alignment_requirement() const {
    return 1;
}

MEGDNN_SPECIALIZE_CREATE_OPERATOR(ChecksumForward);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace cambricon
}  // namespace megdnn

MEGDNN_VERSION_SYMBOL3(CNRT, CNRT_MAJOR_VERSION, CNRT_MINOR_VERSION,
                       CNRT_PATCH_VERSION);

// vim: syntax=cpp.doxygen

