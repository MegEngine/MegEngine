/**
 * \file dnn/src/cambricon/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megcore_cambricon.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/handle_impl.h"
#include "src/common/utils.h"

#include <atomic>
#include <mutex>

#include <cnrt.h>

namespace megdnn {
namespace cambricon {

class HandleImpl : public HandleImplHelper {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle);
    ~HandleImpl() noexcept;

    size_t alignment_requirement() const override;

    const cnrtDeviceInfo_t& device_info() const { return m_device_info; }

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    const megcore::CambriconContext& megcore_context() const {
        return m_megcore_context;
    }

    int device_id() const { return m_device_id; }

    cnrtQueue_t queue() const { return megcore_context().queue; }

    //! global matmul opr
    Checksum* checksum_opr() override final {
        return get_helper_opr<Checksum, 0>(this);
    }

private:
    int m_device_id;
    //! MegDNN handle does not manage the lifetime of cnrt queue.
    megcore::CambriconContext m_megcore_context;

    cnrtDeviceInfo_t m_device_info;
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen

