/**
 * \file src/serialization/include/megbrain/serialization/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/dtype.h"
#include "megbrain/utils/arith_helper.h"
#include "megdnn/thin/function.h"
#include "megdnn/opr_param_defs.h"

namespace mgb {
namespace serialization {

void serialize_dtype(DType dtype,
                     megdnn::thin_function<void(const void*, size_t)> write_fn);
DType deserialize_dtype(megdnn::thin_function<void(void*, size_t)> read_fn);

/*!
 * \brief a compressed encoding for unsigned integers with small values
 * \tparam T the uint type to be encoded/decoded
 */
template <typename T>
class CompressedUint {
    static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value,
                  "T must be uint");

public:
    /*!
     * \brief encode a uint value
     *
     * \param writer a callback to write data; signature: (data, size). It would
     *      only be called once
     */
    template <class Writer>
    static void write(T val, Writer writer) {
        if (!val) {
            int v = 0;
            writer(&v, 1);
            return;
        }

        uint8_t parts_buf[divup<size_t>(sizeof(T) * 8, 7)];
        auto parts_end = parts_buf + sizeof(parts_buf), parts_ptr = parts_end;

        while (val) {
            --parts_ptr;

            *parts_ptr = val & 0x7F;
            if (parts_ptr + 1 < parts_end)
                *parts_ptr |= 0x80;

            val >>= 7;
        }
        mgb_assert(parts_ptr >= parts_buf);
        writer(parts_ptr, parts_end - parts_ptr);
    }

    /*!
     * \brief decode a uint value
     * \param reader a function to read a single byte; it must return uint8_t
     */
    template <class Reader>
    static T read(Reader reader) {
        static_assert(std::is_same<std::remove_cv_t<decltype(reader())>,
                                   uint8_t>::value,
                      "reader must return uint8_t");
        T val = 0;
        for (;;) {
            uint8_t cur = reader();
            val = (val << 7) | (cur & 0x7F);
            if (!(cur & 0x80))
                break;
        }
        return val;
    }
};

}  // namespace serialization
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
