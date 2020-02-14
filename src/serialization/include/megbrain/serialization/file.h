/**
 * \file src/serialization/include/megbrain/serialization/file.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"

namespace mgb {
namespace serialization {

class SharedBuffer {
    std::shared_ptr<const void> m_buf;
    size_t m_size;

public:
    SharedBuffer(std::shared_ptr<const void> buf, size_t size)
            : m_buf{std::move(buf)}, m_size{size} {}

    ~SharedBuffer();

    const void* data() const { return m_buf.get(); }

    size_t size() const { return m_size; }
};

//! abstract input file interface
class InputFile {
    class FsImpl;
    class MemProxyImpl;
    class SharedMemProxyImpl;

public:
    virtual ~InputFile() = default;

    //! reset to beginning of input stream
    virtual void rewind() = 0;

    //! skip given number of bytes
    virtual void skip(size_t bytes) = 0;

    //! read data into buffer
    virtual void read(void* dst, size_t size) = 0;

    //! return current read offset
    virtual size_t tell() = 0;

    /*!
     * \brief read into a host tensor
     *
     * The default implementation uses read(); an alternative
     * implementation might directly reset the storage of \p dest to
     * utilize zero-copy.
     */
    virtual void read_into_tensor(HostTensorND& dest,
                                  const TensorLayout& layout);

    /*!
     * \brief read with sharing memory (i.e. use zero-copy if possible)
     *
     * The default implementation allocates a new buffer and call
     * read().
     *
     * Note that there is no alignment guarantee.
     */
    virtual SharedBuffer read_shared(size_t size);

    //! create an InputFile correspoding to a file on local file system
    static std::unique_ptr<InputFile> make_fs(const char* path);

    //! create an InputFile correspoding to a memory region; the memory
    //! region must be alive throughout lifespan of this InputFile
    static std::unique_ptr<InputFile> make_mem_proxy(const void* ptr,
                                                     size_t size);

    /*!
     * \brief create an InputFile that would directly reuse the memory
     *      buffer to load tensor values
     *
     * \param writable whether the input memory region can be modified.
     *      If this is set to true, tensor storage can be aggressively
     *      shared by reusing the buffer for alignment.
     */
    static std::unique_ptr<InputFile> make_mem_proxy(std::shared_ptr<void> ptr,
                                                     size_t size,
                                                     bool writable = true);
};

//! abstract output file interface
class OutputFile {
    class FsImpl;
    class VectorProxyImpl;

public:
    virtual ~OutputFile() = default;

    //! write buffer to file
    virtual void write(const void* src, size_t size) = 0;

    //! seek to absolute position in bytes
    virtual void seek(size_t offset) = 0;

    //! return current write offset
    virtual size_t tell() = 0;

    //! create an OutputFile correspoding to a file on local file system
    static std::unique_ptr<OutputFile> make_fs(const char* path,
                                               char mode = 'w');

    /*!
     * \brief create an OutputFile to write to a std::vector
     *
     * Note that the vector must be alive throughout lifespan of this
     * OutputFile. Current content in *buf* would not be cleared.
     */
    static std::unique_ptr<OutputFile> make_vector_proxy(
            std::vector<uint8_t>* buf);
};

}  // namespace serialization
}  // namespace mgb