/**
 * \file dnn/test/rocm/checksum.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "test/rocm/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(ROCM, CHECKSUM_FORWARD) {
    auto rocm_opr = handle_rocm()->create_operator<megdnn::Checksum>(),
         naive_opr = handle_naive()->create_operator<megdnn::Checksum>();
    std::mt19937 rng(std::random_device{}());
    for (size_t size: {3, 8, 12345, 1024 * 1024, 1024 * 1024 * 10}) {
        auto aligned_size = size + ((512 - size % 512) % 512);
        auto run = [&](megdnn::Checksum *opr, void *ptr, bool log_size) {
            TensorND tensor;
            tensor.raw_ptr = ptr;
            tensor.layout.init_contiguous_stride({size});
            tensor.layout.dtype = dtype::Byte();
            WorkspaceWrapper workspace(handle_rocm(),
                    opr->get_workspace_in_bytes(tensor.layout));
            if (log_size) {
                printf("checksum(%zu): workspace=%zu\n", size,
                        workspace.workspace().size);
            }
            return opr->exec(tensor, workspace.workspace());
        };
        std::vector<uint8_t> buf(aligned_size);
        for (size_t i = 0; i < size; ++ i)
            buf[i] = rng();
        auto run_offsset = [&](size_t offset) {
            void* dev_ptr = megdnn_malloc(handle_rocm(), buf.size() + offset);
            void* dev_buf = static_cast<char*>(dev_ptr) + offset;

            Checksum::Result res_rocm[2], res_naive[2];

            for (int change_last = 0; change_last < 2; ++ change_last) {
                if (change_last)
                    ++ buf[size - 1];

                megdnn_memcpy_H2D(handle_rocm(), dev_buf, buf.data(), size);
                res_rocm[change_last] = run(rocm_opr.get(), dev_buf, !change_last);
                res_naive[change_last] = run(naive_opr.get(), buf.data(), false);
            }

            megdnn_free(handle_rocm(), dev_ptr);

            ASSERT_EQ(res_naive[0], res_rocm[0]) << "failed for size " << size;
            ASSERT_EQ(res_naive[1], res_rocm[1]);
            ASSERT_NE(res_rocm[0], res_rocm[1]);
        };

        for (size_t i = 0; i < 8; ++i) {
            run_offsset(i);
        }
    }
}

// vim: syntax=cpp.doxygen
