/**
 * \file src/opr/include/tensor_gen.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/tensor.h"

namespace mgb {
/*!
 * A static class including methods to generate host tensors.
 */
class TensorGen {
public:
    /*!
     * \brief Generate a tensor with all the elements equal to the given value
     */
    template <typename ctype, typename = typename mgb::ctype_enable_if<ctype>::type>
    static std::shared_ptr<mgb::HostTensorND> constant(
            mgb::TensorShape shape, ctype value,
            mgb::CompNode comp_node = mgb::CompNode::load("xpu0")) {
        std::shared_ptr<mgb::HostTensorND> r = std::make_shared<mgb::HostTensorND>(
                comp_node, shape, typename mgb::DTypeTrait<ctype>::dtype());
        auto ptr = r->ptr<ctype>();
        for (size_t i = 0, it = r->layout().total_nr_elems(); i < it; i++) {
            ptr[i] = value;
        }

        return r;
    }

    /*!
     * \brief Generate a tensor with all the elements equal to 0
     */
    template <typename T>
    static std::shared_ptr<mgb::HostTensorND> zeros(
            mgb::TensorShape shape,
            mgb::CompNode comp_node = mgb::CompNode::load("xpu0")) {
        static_assert(
                std::is_base_of<mgb::DType, T>(),
                "Please use the dtype in namespace mgb or use "
                "Tensor::constant.");
        using ctype = typename mgb::DTypeTrait<T>::ctype;
        return constant(shape, (ctype)0, comp_node);
    }

    /*!
     * \brief Generate a tensor with all the elements equal to 0. In this method
     * typename is not required.
     */
    static std::shared_ptr<mgb::HostTensorND> zeros(
            mgb::TensorShape shape, mgb::DType dtype = mgb::dtype::Float32(),
            mgb::CompNode comp_node = mgb::CompNode::load("xpu0")) {
        std::shared_ptr<mgb::HostTensorND> r =
                std::make_shared<mgb::HostTensorND>(comp_node, shape, dtype);
        auto ptr = r->raw_ptr();
        memset(ptr, 0, sizeof(megdnn::dt_byte));
        return r;
    }

    /*!
     * \brief Generate a tensor with all the elements equal to 1
     */
    template <typename T>
    static std::shared_ptr<mgb::HostTensorND> ones(
            mgb::TensorShape shape,
            mgb::CompNode comp_node = mgb::CompNode::load("xpu0")) {
        static_assert(
                std::is_base_of<mgb::DType, T>(),
                "Please use the dtype in namespace mgb or use "
                "Tensor::constant.");
        using ctype = typename mgb::DTypeTrait<T>::ctype;
        return constant(shape, (ctype)1, comp_node);
    }
};
}  // namespace mgb
