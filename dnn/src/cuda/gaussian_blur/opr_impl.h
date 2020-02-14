/**
 * \file dnn/src/cuda/gaussian_blur/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"

#include "src/common/utils.h"
#include "src/common/cv/common.h"
#include <cstring>

namespace megdnn {
namespace cuda {

class GaussianBlurImpl : public GaussianBlur {
    public:
        using GaussianBlur::GaussianBlur;

        void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                  _megdnn_workspace workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout& src,
                                      const TensorLayout&) override {
            //! current only support float and uint8
            megdnn_assert(src.dtype == dtype::Float32() ||
                          src.dtype == dtype::Uint8());

            //! Calc gaussian kernel real size
            double sigma_x = param().sigma_x;
            double sigma_y = param().sigma_y;
            uint32_t kernel_height = param().kernel_height;
            uint32_t kernel_width = param().kernel_width;

            if (sigma_y <= 0)
                sigma_y = sigma_x;

            auto get_size = [&src](double sigma) {
                double num = 0;
                if (src.dtype == dtype::Uint8()) {
                    num = sigma * 3 * 2 + 1;
                } else {
                    num = sigma * 4 * 2 + 1;
                }
                return static_cast<uint32_t>(num + (num >= 0 ? 0.5 : -0.5)) | 1;
            };

            if (kernel_width <= 0 && sigma_x > 0) {
                m_kernel_width = get_size(sigma_x);
            } else {
                m_kernel_width = kernel_width;
            }
            if (kernel_height <= 0 && sigma_y > 0) {
                m_kernel_height = get_size(sigma_y);
            } else {
                m_kernel_height = kernel_height;
            }
            megdnn_assert(m_kernel_width > 0 && m_kernel_width % 2 == 1 &&
                          m_kernel_height > 0 && m_kernel_height % 2 == 1);

            m_sigma_x = std::max(sigma_x, 0.);
            m_sigma_y = std::max(sigma_y, 0.);

            if (src.dtype == dtype::Uint8()) {
                //! element [0, m_kernel_width * m_kernel_height - 1] store the
                //! filter matrix of type int32_t, others store float value
                //! kernel_x and kernel_y.
                return m_kernel_width * m_kernel_height * sizeof(int32_t) +
                       (m_kernel_width + m_kernel_height) * sizeof(float);
            } else {
                //! float32
                return m_kernel_width * m_kernel_height * sizeof(float);
            }
        }

    private:
        uint32_t m_kernel_height;
        uint32_t m_kernel_width;
        double m_sigma_x;
        double m_sigma_y;

};  // class GaussianBlurImpl

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
