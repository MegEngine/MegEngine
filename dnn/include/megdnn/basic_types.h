/**
 * \file dnn/include/megdnn/basic_types.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/arch.h"
#include "megdnn/dtype.h"
#include "megdnn/internal/defs.h"

#include <memory>

#if MEGDNN_CC_HOST
#include <cstdarg>
#include <string>
#include <type_traits>
#include <vector>
#include "megdnn/thin/small_vector.h"
#endif  // MEGDNN_CC_HOST

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {

class ErrorHandler {
#if MEGDNN_CC_HOST
    static ErrorHandler* sm_inst;
    static ErrorHandler* inst();

protected:
    MEGDNN_NORETURN virtual void do_on_megdnn_error(const std::string& msg) = 0;

    MEGDNN_NORETURN virtual void do_on_tensor_reshape_error(const std::string& msg) {
        on_megdnn_error(msg);
    }

    ~ErrorHandler() = default;

#endif
public:
    //! called on general megdnn error
    MEGDNN_NORETURN static void on_megdnn_error(const char* msg);

    //! called on tensor reshape error
    MEGDNN_NORETURN static void on_tensor_reshape_error(const char* msg);

#if MEGDNN_CC_HOST
    MEGDNN_NORETURN static void on_megdnn_error(const std::string& msg);
    MEGDNN_NORETURN static void on_tensor_reshape_error(const std::string& msg);

    /*!
     * \brief set the global error handler instance
     *
     * This method is not thread-safe. The caller is responsible to ensure the
     * ErrorHandler is a global object with enough life span.
     *
     * \return original error handler
     */
    static void set_handler(ErrorHandler* handler);

#endif  // MEGDNN_CC_HOST
};

#if MEGDNN_CC_HOST
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

typedef void (*LogHandler)(
        LogLevel level, const char* file, const char* func, int line, const char* fmt,
        va_list ap);

/*!
 * \brief set the callback to receive all log messages
 *
 * Note: the log handler can be NULL (which is also the default value). In this
 * case, no log message would be recorded.
 *
 * \return original log handler
 */
LogHandler set_log_handler(LogHandler handler);
#endif

/**
 * \brief Describing the tensor shape.
 *
 * Uninitialized shape: ndim == 0; total_nr_elems() is also defined to be 0
 *
 * Empty shape: ndim > 0 && shape[i] == 0 for 0 <= i < ndim; it is always
 * considered non-contiguous.
 */
struct TensorShape {
    static MEGDNN_CONSTEXPR size_t MAX_NDIM = MEGDNN_MAX_NDIM;

#if MEGDNN_CC_HOST
    size_t shape[MAX_NDIM], ndim = 0;
#else
    size_t shape[MAX_NDIM], ndim;
#endif

#if MEGDNN_CC_HOST
    TensorShape() = default;
    TensorShape(const TensorShape& rhs) = default;
    MGE_WIN_DECLSPEC_FUC TensorShape(const SmallVector<size_t>& init_shape);
    MGE_WIN_DECLSPEC_FUC TensorShape(std::initializer_list<size_t> init_shape);
    MGE_WIN_DECLSPEC_FUC std::string to_string() const;
#endif

    //! total number of elements
    MGE_WIN_DECLSPEC_FUC size_t total_nr_elems() const;

    //! check whether two shapes are equal
    MGE_WIN_DECLSPEC_FUC bool eq_shape(const TensorShape& rhs) const;

    //! check whether the shape can be treated as a scalar
    bool is_scalar() const { return ndim == 1 && shape[0] == 1; }

    //! check whether ndim != 0 and at least one shape is 0
    MGE_WIN_DECLSPEC_FUC bool is_empty() const;

    //! access single element, without boundary check
    size_t& operator[](size_t i) { return shape[i]; }
    size_t operator[](size_t i) const { return shape[i]; }
};

class Handle;
/**
 * \brief Describing the tensor shape with its actual layout in memory and dtype
 *
 * x(i, j, ...) is stored at offset
 * stride[0]*i + stride[1]*j + ..., in number of elements; physical offset needs
 * to be multiplied by dtype size.
 */
struct TensorLayout : public TensorShape {
    /*!
     * \brief Describes min and max offsets of tensor elements with respect to
     *      its first element, so all tensor elements are guaranteed to be in
     *      the range [elem[0]+low, elem[0]+last). Besides, we have a high to
     *      describe the range including row pitch when using image2D
     */
    struct Span {
        ptrdiff_t low_elem, low_byte;
        size_t high_elem, high_byte;
        //! The differece between high_elem and last elem is that last_elem describes
        //! the last element of a tensor regardless of the row pitch at the last row. It
        //! will be useful when copying into a part of image.
        size_t last_elem, last_byte;

        Span(ptrdiff_t low_elem, ptrdiff_t low_byte, size_t high_elem, size_t high_byte,
             size_t last_elem, size_t last_byte)
                : low_elem(low_elem),
                  low_byte(low_byte),
                  high_elem(high_elem),
                  high_byte(high_byte),
                  last_elem(last_elem),
                  last_byte(last_byte) {}
        size_t dist_elem() const { return high_elem - low_elem; }

        size_t dist_byte() const { return high_byte - low_byte; }

        size_t dist_last_byte() const { return last_byte - low_byte; }
    };

    /*!
     * \brief Describing the requirements for tensor layouts
     *
     * Some runtime (e.g. opencl) may have alignment requirements for special
     * memory types (e.g. image in texture memory). Format objects can be used
     * to impose such constraints on methods related to tensor strides.
     *
     * Note that ImplBase is defined in tensor_format.h
     */
    class Format {
    public:
        class ImplBase;

#if MEGDNN_CC_HOST
        MGE_WIN_DECLSPEC_FUC Format();
        MGE_WIN_DECLSPEC_FUC Format(DType dtype);

        const ImplBase* impl() const { return m_impl; }

        enum class Type;

        //! get impl type; defined in tensor_format.h
        inline Type type() const;

        //! convert to the implementation class; exception would be raised if
        //! type mismatches
        template <class Impl>
        const Impl& as_impl() const {
            static_assert(std::is_base_of<ImplBase, Impl>::value, "bad type");
            if (type() != Impl::TYPE) {
                on_bad_cvt(Impl::TYPE);
            }
            return *static_cast<const Impl*>(m_impl);
        }

        //! get human-readable string description of this format
        MGE_WIN_DECLSPEC_FUC std::string to_string() const;

        MGE_WIN_DECLSPEC_FUC std::string serialize() const;
        MGE_WIN_DECLSPEC_FUC static Format deserialize(
                const std::string& bin, const Handle* handle);

        //! whether this is the default tensor format
        MGE_WIN_DECLSPEC_FUC bool is_default() const;

        //! whether this is the lowbit aligned to bytes tensor format
        MGE_WIN_DECLSPEC_FUC bool is_lowbit_aligned() const;

        bool operator==(Format rhs) const { return m_impl == rhs.m_impl; }
        bool operator!=(Format rhs) const { return m_impl != rhs.m_impl; }
#endif

    private:
        const ImplBase* m_impl;

#if MEGDNN_CC_HOST
        Format(ImplBase* impl) : m_impl{impl} {}
        MEGDNN_NORETURN void on_bad_cvt(Type dst_type) const;
#endif
    };

    ptrdiff_t stride[MAX_NDIM];
    DType dtype;
    Format format;

    MGE_WIN_DECLSPEC_FUC TensorLayout();

#if MEGDNN_CC_HOST
    TensorLayout(const TensorLayout& layout) = default;

    //! create empty layout with given dtype
    MGE_WIN_DECLSPEC_FUC explicit TensorLayout(DType dtype_);

    MGE_WIN_DECLSPEC_FUC TensorLayout(DType dtype_, Format format);

    //! create layout with given shape and contiguous stride.
    MGE_WIN_DECLSPEC_FUC TensorLayout(const TensorShape& shape, DType dtype);

    MGE_WIN_DECLSPEC_FUC TensorLayout(
            const TensorShape& shape, DType dtype, Format format);

    //! creating layout with user-specified shape and stride.
    MGE_WIN_DECLSPEC_FUC TensorLayout(
            const TensorShape& shape, const std::vector<ptrdiff_t>& stride,
            DType dtype);

    MGE_WIN_DECLSPEC_FUC TensorLayout(
            const TensorShape& shape, const std::vector<ptrdiff_t>& stride, DType dtype,
            Format format);

    /* =================== inplace modifiers =================== */

    /*!
     * \brief init stride to be contiguous
     *
     * Use current shape and format
     *
     * \return total number of elements
     */
    MGE_WIN_DECLSPEC_FUC size_t init_contiguous_stride();

    /*!
     * \brief init stride to be contiguous by first assigning shape
     *
     * Use current format.
     */
    MGE_WIN_DECLSPEC_FUC size_t init_contiguous_stride(const TensorShape& shape);

    MGE_WIN_DECLSPEC_FUC size_t
    init_contiguous_stride(const TensorShape& shape, Format format);

    /*!
     * \brief inplace version of remove_axis
     */
    MGE_WIN_DECLSPEC_FUC void remove_axis_inplace(size_t idx);

    /*!
     * \brief add an axis before given *axis* with given shape and stride
     *
     * Other shapes and strides would not be changed.
     */
    MGE_WIN_DECLSPEC_FUC void add_axis_inplace(
            size_t axis, size_t shape, ptrdiff_t stride);

    /*!
     * \brief add an axis before given *axis*, with shape 1 and contiguous
     *      stride
     */
    void add_axis_cont_inplace(size_t axis) {
        ptrdiff_t stride_ = axis < ndim ? stride[axis] * shape[axis] : 1;
        add_axis_inplace(axis, 1, stride_);
    }

    /*!
     * \brief modify data type of the layout inplace
     *
     * By the way this API will modify the format according to the data type
     */
    MGE_WIN_DECLSPEC_FUC void modify_dtype_inplace(DType dtype);

    /* =================== generate new layout =================== */

    /**
     * \brief Returns the layout with permuted dimensions.
     *
     * example:
     *  (2, 0, 1) -> AxBxC to CxAxB
     */
    MGE_WIN_DECLSPEC_FUC TensorLayout
    dimshuffle(const std::vector<size_t>& dims) const MEGDNN_WARN_UNUSED_RESULT;

    /**
     * \brief Remove an axis from the layout by moving later shape/stride
     *      elements earlier. No extra check is performed.
     */
    MGE_WIN_DECLSPEC_FUC TensorLayout
    remove_axis(size_t idx) const MEGDNN_WARN_UNUSED_RESULT;

    /**
     * \brief Returns a different view.
     *
     * \throw TensorReshapeError if no stride exists for target shape.
     */
    MGE_WIN_DECLSPEC_FUC TensorLayout
    reshape(const TensorShape& shape) const MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief try to reshape to another view; return whether these two shapes
     *      are compatible
     * \return true iff there exists target stride so this layout can be
     *      converted to target shape and the elements can match.
     */
    MGE_WIN_DECLSPEC_FUC bool try_reshape(
            TensorLayout& output,
            const TensorShape& shape) const MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief Broadcast on dims with shape == 1 to match target *shape*.
     * \throw TensorReshapeError if could not be satisfied
     */
    MGE_WIN_DECLSPEC_FUC TensorLayout
    broadcast(const TensorShape& shape) const MEGDNN_WARN_UNUSED_RESULT;

    /*!
     * \brief Collapse consecutive axes with contiguous layout together
     *
     * This transforms the tensor into a canonized form. For empty tensors or
     * scalar, the result would always be a one-dimensional empty or scalar,
     * with stride being 1.
     */
    MGE_WIN_DECLSPEC_FUC TensorLayout
    collapse_contiguous() const MEGDNN_WARN_UNUSED_RESULT;

    /* =================== properties =================== */

    MGE_WIN_DECLSPEC_FUC std::string to_string() const;

    MGE_WIN_DECLSPEC_FUC std::string serialize() const;
#endif  // MEGDNN_CC_HOST

    /*!
     * \brief check whether the is contiguous under its format definition
     *
     * See is_contiguous_spec() in Format impl classes for more detail. When the
     * format is default, this is equivalent to is_physical_contiguous().
     *
     * Note that empty tensors (i.e. with 0 shapes) are not considered as
     * contiguous.
     */
    MGE_WIN_DECLSPEC_FUC bool is_contiguous() const;

    //! check whether it is physically contiguous disregarding format
    MGE_WIN_DECLSPEC_FUC bool is_physical_contiguous() const;

    /*!
     * \brief check whether the layout is monotonous
     *
     * A tensor is monotonous if abs(stride[i]) >= abs(stride[i+1])*shape[i+1]
     */
    MGE_WIN_DECLSPEC_FUC bool is_abs_monotonous_allow_brdcst() const;

    /*!
     * \brief check whether the layout is contiguous, allowing broadcasting
     *
     * This checks whether the underlying storage is contiguous, where
     * broadcasting is also considered to be so.
     */
    MGE_WIN_DECLSPEC_FUC bool is_contiguous_allow_brdcst() const;

    /*!
     * \brief if this function returns true, then no two elements can occupy the
     *      same memory slot
     *
     * Note that this test is a sufficient but not necessary condition for the
     * layout being non-overlapping: when this function returns false, it is
     * still possible that actually no two elements share the same memory
     * location.
     */
    MGE_WIN_DECLSPEC_FUC bool is_non_overlapping_strong() const;

    MGE_WIN_DECLSPEC_FUC bool eq_layout(const TensorLayout& rhs) const;

    //! get lowest and highest offset reachable from this layout
    MGE_WIN_DECLSPEC_FUC Span span() const;

    //! total number of access bytes
    MGE_WIN_DECLSPEC_FUC size_t access_bytes() const;
};

class RefPtr {
    std::shared_ptr<void*> m_ref;
    size_t m_offset;
    bool m_mutable;

public:
    RefPtr() {
        m_ref = std::make_shared<void*>((void*)nullptr);
        m_offset = 0;
        m_mutable = true;
    }

    RefPtr(void* ref_ptr, const size_t offset = 0) {
        m_ref = std::make_shared<void*>(ref_ptr);
        m_offset = offset;
        m_mutable = true;
    }

    explicit RefPtr(
            std::shared_ptr<void*> ref_ptr, const size_t offset = 0,
            bool is_mutable = true) {
        m_ref = ref_ptr;
        m_offset = offset;
        m_mutable = is_mutable;
    }

    void* get_ptr() const {
        return static_cast<void*>(
                (*m_ref != NULL) ? static_cast<dt_byte*>(*m_ref) + m_offset : nullptr);
    }

    bool is_mutable() const { return m_mutable; }

    void reset(const void* ptr, size_t offset = 0);

    RefPtr& operator+=(size_t offset) {
        m_offset += offset;
        return *this;
    }

    bool operator==(const RefPtr& other) const {
        return *m_ref == *other.m_ref && m_offset == other.m_offset;
    }

    template <typename T>
    T* ptr() const {
        return static_cast<T*>(get_ptr());
    }
};

/**
 * \brief A simple encapsulation class for n-dimensional tensor.
 */
struct TensorND {
    TensorLayout layout;

    TensorND() : m_ref_ptr(RefPtr((void*)nullptr)) {}

    TensorND(void* raw_ptr_, const TensorLayout& layout_)
            : layout(layout_), m_ref_ptr(raw_ptr_) {}

    TensorND(const TensorLayout& layout_, const RefPtr& ref_ptr)
            : layout(layout_), m_ref_ptr(ref_ptr) {}

    MGE_WIN_DECLSPEC_FUC void reset_ptr(void* ptr, size_t offset = 0);

    void* raw_ptr() const { return m_ref_ptr.get_ptr(); }

    const RefPtr get_ref_ptr() const { return m_ref_ptr; }

    RefPtr& get_ref_ptr() { return m_ref_ptr; }

    //! get typed pointer; type check is performed
    template <typename T>
    T* ptr() const {
        layout.dtype.assert_is_ctype<T>();
        return static_cast<T*>(m_ref_ptr.get_ptr());
    }

    //! get typed pointer of compatible type
    template <typename T>
    T* compatible_ptr() const {
        layout.dtype.assert_is_compatible_ctype<T>();
        return reinterpret_cast<T*>(m_ref_ptr.get_ptr());
    }

private:
    RefPtr m_ref_ptr;
};

#if MEGDNN_CC_HOST
using TensorFormat = TensorLayout::Format;
using TensorShapeArray = SmallVector<TensorShape>;
using TensorNDArray = SmallVector<TensorND>;
using TensorLayoutArray = SmallVector<TensorLayout>;
using TensorLayoutPtrArray = SmallVector<TensorLayout*>;
using TensorFormatArray = SmallVector<TensorFormat>;
#endif

/**
 * \brief A struct representing workspace.
 *
 * It differs from TensorND in that workspace does not have a "layout" concept.
 */
struct Workspace {
    dt_byte* raw_ptr;
    size_t size;

    Workspace() : raw_ptr(NULL), size(0) {}

    Workspace(dt_byte* raw_ptr_, size_t size_) : raw_ptr(raw_ptr_), size(size_) {}

    template <typename T>
    T* ptr(size_t offset_in_bytes = 0) const {
        return static_cast<T*>(static_cast<void*>(raw_ptr + offset_in_bytes));
    }
};

#if MEGDNN_CC_HOST

/*!
 * \brief manage output and workspace memory for dynamic output oprs
 */
class DynOutMallocPolicy {
protected:
    ~DynOutMallocPolicy() = default;

public:
    /*!
     * \brief allocate an output var
     * \param id output index, starting from 0
     * \param dtype requested output data type
     * \param shape requested output shape
     * \param user_data extra user data passed in DynOutMallocPolicyCall
     */
    virtual TensorND alloc_output(
            size_t id, DType dtype, const TensorShape& shape, void* user_data) = 0;

    /*!
     * \brief allocate workspace memory
     * \param sz requested workspace in bytes
     */
    virtual void* alloc_workspace(size_t sz, void* user_data) = 0;

    /*!
     * \brief free workspace memory
     *
     * Every operator should guarantee that alloc_workspace() and
     * free_workspace() calls are matched
     */
    virtual void free_workspace(void* ptr, void* user_data) = 0;
};

/*!
 * \brief bind a DynOutMallocPolicy with arbitrary user data
 */
struct DynOutMallocPolicyCall {
    DynOutMallocPolicy* policy;
    void* user_data;

    DynOutMallocPolicyCall(DynOutMallocPolicy* p = nullptr, void* ud = nullptr)
            : policy{p}, user_data{ud} {}

    TensorND alloc_output(size_t id, DType dtype, const TensorShape& shape) {
        return policy->alloc_output(id, dtype, shape, user_data);
    }

    /*!
     * \brief allocate workspace with return type conversion
     * \tparam elem element type for size calculation
     * \param nr_elem number of elements; allocated size is sizeof(elem) *
     *      nr_elem
     */
    template <typename T = void, typename elem = T>
    T* alloc_workspace(size_t nr_elem) {
        using real_elem = typename std::conditional<
                std::is_same<elem, void>::value, uint8_t, elem>::type;
        return static_cast<T*>(
                policy->alloc_workspace(nr_elem * sizeof(real_elem), user_data));
    }

    void free_workspace(void* ptr) { return policy->free_workspace(ptr, user_data); }
};

template <typename T>
class EnumClassBit {
    std::underlying_type_t<T> m_val;

    constexpr EnumClassBit(std::underlying_type_t<T> v) : m_val(v) {}

public:
    constexpr EnumClassBit(T v) : m_val(static_cast<std::underlying_type_t<T>>(v)) {}

    constexpr operator T() const { return static_cast<T>(m_val); }

    constexpr explicit operator bool() const { return m_val; }

#define DEF_OPR(op)                                                     \
    constexpr EnumClassBit operator op(const EnumClassBit& rhs) const { \
        return m_val op rhs.m_val;                                      \
    }

    DEF_OPR(&)
    DEF_OPR(|)
    DEF_OPR(^)

    constexpr EnumClassBit operator~() const { return ~m_val; }

#undef DEF_OPR
};

#endif  // MEGDNN_CC_HOST

}  // namespace megdnn

#define _MEGDNN_DECBO_SINGLE_OPR(cls, op)                                        \
    inline constexpr ::megdnn::EnumClassBit<cls> operator op(cls x, cls y) {     \
        return ::megdnn::EnumClassBit<cls>(x) op ::megdnn::EnumClassBit<cls>(y); \
    }                                                                            \
    inline constexpr ::megdnn::EnumClassBit<cls> operator op(                    \
            ::megdnn::EnumClassBit<cls> x, cls y) {                              \
        return x op ::megdnn::EnumClassBit<cls>(y);                              \
    }

#define _MEGDNN_DECBO_SINGLE_OPR_ASSIGN(cls, op)          \
    inline constexpr cls& operator op##=(cls& x, cls y) { \
        x = x op ::megdnn::EnumClassBit<cls>(y);          \
        return x;                                         \
    }

#define MEGDNN_DEF_ENUM_CLASS_BIT_OPR(cls)                          \
    _MEGDNN_DECBO_SINGLE_OPR(cls, &)                                \
    _MEGDNN_DECBO_SINGLE_OPR(cls, |)                                \
    _MEGDNN_DECBO_SINGLE_OPR(cls, ^)                                \
    _MEGDNN_DECBO_SINGLE_OPR_ASSIGN(cls, &)                         \
    _MEGDNN_DECBO_SINGLE_OPR_ASSIGN(cls, |)                         \
    _MEGDNN_DECBO_SINGLE_OPR_ASSIGN(cls, ^)                         \
    inline constexpr ::megdnn::EnumClassBit<cls> operator~(cls x) { \
        return ~::megdnn::EnumClassBit<cls>(x);                     \
    }

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
