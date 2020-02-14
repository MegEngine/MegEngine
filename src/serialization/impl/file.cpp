/**
 * \file src/serialization/impl/file.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/serialization/file.h"

namespace mgb {
namespace serialization {

SharedBuffer::~SharedBuffer() = default;

/* ====================== InputFile ====================== */
void InputFile::read_into_tensor(HostTensorND& dest,
                                 const TensorLayout& layout) {
    dest.dtype(layout.dtype).resize(layout);
    read(dest.raw_ptr(), layout.span().high_byte);
}

SharedBuffer InputFile::read_shared(size_t size) {
    std::shared_ptr<void> shptr{new uint8_t[size],
                                [](uint8_t* p) { delete[] p; }};
    read(shptr.get(), size);
    return {std::move(shptr), size};
}

/* ====================== file impls ====================== */
class InputFile::FsImpl final : public InputFile {
    FILE* m_fptr;

public:
    FsImpl(const char* path) : m_fptr{fopen(path, "rb")} {
        mgb_assert(m_fptr, "failed to open %s: %s", path, strerror(errno));
    }

    ~FsImpl() {
        if (m_fptr)
            fclose(m_fptr);
    }

    void rewind() override { std::rewind(m_fptr); }

    void skip(size_t bytes) override {
        auto err = fseek(m_fptr, bytes, SEEK_CUR);
        mgb_assert(!err);
    }

    void read(void* dst, size_t size) override {
        auto nr = fread(dst, 1, size, m_fptr);
        mgb_assert(nr == size);
    }

    size_t tell() override { return std::ftell(m_fptr); }
};

std::unique_ptr<InputFile> InputFile::make_fs(const char* path) {
    return std::make_unique<FsImpl>(path);
}

class OutputFile::FsImpl final : public OutputFile {
    FILE* m_fptr;

public:
    FsImpl(const char* path, char mode) {
        mgb_assert(mode == 'w' || mode == 'a', "invalid mode: %c", mode);
        m_fptr = fopen(path, mode == 'w' ? "wb" : "r+b");
        mgb_assert(m_fptr, "failed to open %s: %s", path, strerror(errno));
        if (mode == 'a') {
            auto err = fseek(m_fptr, 0, SEEK_END);
            mgb_assert(!err, "failed to seek to end");
        }
    }

    ~FsImpl() {
        if (m_fptr)
            fclose(m_fptr);
    }

    void write(const void* src, size_t size) override {
        auto nr = fwrite(src, 1, size, m_fptr);
        mgb_assert(nr == size);
    }

    void seek(size_t offset) override {
        auto err = fseek(m_fptr, offset, SEEK_SET);
        mgb_assert(!err);
    }

    size_t tell() override {
        auto pos = ftell(m_fptr);
        mgb_assert(pos >= 0);
        return pos;
    }
};

std::unique_ptr<OutputFile> OutputFile::make_fs(const char* path, char mode) {
    return std::make_unique<FsImpl>(path, mode);
}

/* ====================== memory impls ====================== */
class InputFile::MemProxyImpl final : public InputFile {
    const uint8_t* const m_ptr;
    size_t const m_size;
    size_t m_offset = 0;

public:
    MemProxyImpl(const void* ptr, size_t size)
            : m_ptr{static_cast<const uint8_t*>(ptr)}, m_size{size} {
        mgb_assert(ptr && size);
    }

    void rewind() override { m_offset = 0; }

    void skip(size_t bytes) override {
        m_offset += bytes;
        mgb_assert(m_offset <= m_size);
    }

    void read(void* dst, size_t size) override {
        mgb_assert(m_offset + size <= m_size);
        memcpy(dst, m_ptr + m_offset, size);
        m_offset += size;
    }

    size_t tell() override { return m_offset; }
};

class InputFile::SharedMemProxyImpl final : public InputFile {
    const bool m_writable;
    bool m_usable = true, m_modified = false;
    std::shared_ptr<void> m_refhold;
    uint8_t* const m_ptr;
    size_t const m_size;
    size_t m_offset = 0;
    //! end of block that is used for tensor value
    //! note we use a signed type to avoid checking ptr underflow
    intptr_t m_write_end = 0;

public:
    SharedMemProxyImpl(std::shared_ptr<void> ptr, size_t size, bool writable)
            : m_writable{writable},
              m_refhold{std::move(ptr)},
              m_ptr{static_cast<uint8_t*>(m_refhold.get())},
              m_size{size} {
        mgb_assert(m_refhold && size);
    }

    void rewind() override {
        if (m_modified) {
            // data has beem modified; can not read again
            m_usable = false;
        }
        m_offset = 0;
    }

    void skip(size_t bytes) override {
        m_offset += bytes;
        mgb_assert(m_offset <= m_size);
    }

    void read(void* dst, size_t size) override {
        mgb_assert(m_usable,
                   "can not read SharedMemProxyImpl again after buf has "
                   "been modified");
        mgb_assert(m_offset + size <= m_size);
        memcpy(dst, m_ptr + m_offset, size);
        m_offset += size;
    }

    size_t tell() override { return m_offset; }

    void read_into_tensor(HostTensorND& dest,
                          const TensorLayout& layout) override;

    SharedBuffer read_shared(size_t size) override;
};

void InputFile::SharedMemProxyImpl::read_into_tensor(
        HostTensorND& dest, const TensorLayout& layout) {
    auto size = layout.span().high_byte;
    mgb_assert(m_offset + size <= m_size);
    void* ptr = m_ptr + m_offset;
    auto align = dest.comp_node().get_mem_addr_alignment();
    auto aligned_write_pos =
            static_cast<intptr_t>(reinterpret_cast<uintptr_t>(ptr) &
                                  ~(align - 1)) -
            reinterpret_cast<intptr_t>(m_ptr);

    void* ptr_to_share = nullptr;
    if (m_writable && size >= align * 4 && aligned_write_pos >= m_write_end) {
        // reuse memory
        void* ptr_aligned = m_ptr + aligned_write_pos;
        if (ptr_aligned != ptr) {
            mgb_assert(ptr_aligned < ptr);
            memmove(ptr_aligned, ptr, size);
            m_modified = true;
        }
        m_write_end = aligned_write_pos + size;
        ptr_to_share = ptr_aligned;
    } else if (!m_writable &&
               !(reinterpret_cast<uintptr_t>(ptr) & (align - 1))) {
        // aligned by chance in read-only mode
        ptr_to_share = ptr;
    }

    if (ptr_to_share) {
        HostTensorStorage storage;
        storage.reset(dest.comp_node(), size,
                      {m_refhold, static_cast<dt_byte*>(ptr_to_share)});
        dest.reset(storage, layout);
    } else {
        // copy to new buffer
        dest.dtype(layout.dtype).resize(layout);
        memcpy(dest.raw_ptr(), ptr, size);
    }
    m_offset += size;
}

SharedBuffer InputFile::SharedMemProxyImpl::read_shared(size_t size) {
    mgb_assert(m_offset + size <= m_size);
    auto ptr = m_ptr + m_offset;
    m_offset += size;
    if (m_writable) {
        mgb_assert(m_offset > static_cast<uintptr_t>(m_write_end));
        m_write_end = m_offset;
    }
    std::shared_ptr<const void> ret{m_refhold, ptr};
    return {std::move(ret), size};
}

std::unique_ptr<InputFile> InputFile::make_mem_proxy(const void* ptr,
                                                     size_t size) {
    return std::make_unique<MemProxyImpl>(ptr, size);
}

std::unique_ptr<InputFile> InputFile::make_mem_proxy(std::shared_ptr<void> ptr,
                                                     size_t size,
                                                     bool writable) {
    return std::make_unique<SharedMemProxyImpl>(std::move(ptr), size, writable);
}

class OutputFile::VectorProxyImpl final : public OutputFile {
    std::vector<uint8_t>* const m_buf;
    size_t m_offset;

public:
    VectorProxyImpl(std::vector<uint8_t>* buf) : m_buf{buf} {
        mgb_assert(buf);
        m_offset = buf->size();
    }

    void write(const void* src, size_t size) override {
        if (m_offset + size > m_buf->size()) {
            m_buf->resize(m_offset + size);
        }
        memcpy(m_buf->data() + m_offset, src, size);
        m_offset += size;
    }

    void seek(size_t offset) override {
        mgb_assert(offset <= m_buf->size());
        m_offset = offset;
    }

    size_t tell() override { return m_offset; }
};

std::unique_ptr<OutputFile> OutputFile::make_vector_proxy(
        std::vector<uint8_t>* buf) {
    return std::make_unique<VectorProxyImpl>(buf);
}

}  // namespace serialization
}  // namespace mgb
