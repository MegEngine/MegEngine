/**
 * \file python_module/src/cpp/megbrain_pubapi.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief public API for exposing megbrain internal data structures
 *
 * This is a pure header without compile-time dependencies.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace mgb {
namespace pubapi {

    /*!
     * \brief a general callback that would be invoked exactly once
     *
     * During the invoke, the functor shoule release related memory
     */
    struct CallbackOnce {
        void (*fptr)(void *);
        void *user_data;

        //! invoke the callback and clean up the scene
        void consume() {
            fptr(user_data);
            fptr = nullptr;
            user_data = nullptr;
        }
    };

    //! tensor on a computing device
    class DeviceTensor {
        public:
            static constexpr uint32_t CURRENT_VERSION = 20190725;

            //! device type
            enum class Type: uint32_t {
                CPU, CUDA
            };
            enum class DataType: uint32_t {
                FLOAT32, FLOAT16, INT32, INT16, INT8, UINT8
            };
            enum class CopyDirection {
                SELF_TO_OTHER, OTHER_TO_SELF
            };
            struct CudaContext {
                int device;     //! set to -1 in copy() to use current device
                void *stream;   //!< set to nullptr for default stream
            };

            //! tensor descriptor
            struct Desc {
                Type type;
                DataType dtype;
                void *dev_ptr;          //!< pointer to actual device buffer
                const size_t *shape;    //!< pointer to shape array
                size_t ndim;
                //! only valid if type == Type::CUDA
                CudaContext cuda_ctx;
            };

            uint32_t _version0; //!< for consistency check
            // note: fields starting with underscore are for internal use only

            Desc desc;
            size_t size_bytes;

            /*!
             * \brief synchonize with the calling thread
             *
             * This must be called before forwarding memory for direct use
             *
             * \param strong whether to synchronoze the whole device (true), or
             *      just the computing node (false). Currently it only affects
             *      how cuda sync is performed.
             */
            void sync(bool strong = false) const {
                m_functable->sync(this, strong);
            }

            /*!
             * \brief copy to/from another buffer
             *
             * Note: the copy is performed on the comp node on which this tensor
             * resides and is always async.
             *
             * If \p direction is OTHER_TO_SELF and shape of this changes, then
             * the corresponding dev_ptr would also be updated.
             *
             * \param other the other buffer involved in the copy; if
             *      \p direction is SELF_TO_OTHER, then only its type and
             *      dev_ptr would be used
             * \param direction specify the direction to perform the copy
             */
            void copy(const Desc &other, CopyDirection direction) {
                m_functable->copy(this, other, direction);
            }

            /*!
             * \brief resize this tensor to given shape
             */
            void resize(size_t ndim, const size_t *shape) {
                Desc tmp;
                tmp.dev_ptr = nullptr;
                tmp.ndim = ndim;
                tmp.shape = shape;
                copy(tmp, CopyDirection::OTHER_TO_SELF);
            }

            //! name of dtype of this tensor
            const char* dtype_name() const { return dtype_name(desc.dtype); }

            //! name of given dtype
            const char* dtype_name(DataType dtype) const {
                return m_functable->dtype_name(dtype);
            }

            /*!
             * \brief forward memory from \p other directly to the underlying
             *      storage
             *
             * This can only be used when there is a corresponding VarNode for
             * this DeviceTensor. (e.g. for the outputs of Craniotome oprs)
             */
            void forward_other_memory(
                    const Desc &other, CallbackOnce deleter) const {
                m_functable->forward_other_memory(this, other, deleter);
            }

            /*!
             * \brief forward device buffer to \p dest directly and create a
             * tensor storage shared memory with m_dv_nd, it would be deleted
             * when calling deleter, so refcnt to data ptr could be managed
             * correctly.
             */
            void forward_to(
                    void **dest, CallbackOnce* deleter) const {
                m_functable->forward_to(this, dest, deleter);
            }

            struct _Impl;
        private:
            // note: we use a func table to avoid symbol visibility problems and
            // linking hazards when built with other code base
            struct FuncTable {
                void (*sync)(const DeviceTensor*, bool);
                void (*copy)(DeviceTensor*, const Desc&, CopyDirection);
                void (*forward_other_memory)(const DeviceTensor*, const Desc&,
                                             CallbackOnce);
                const char* (*dtype_name)(DataType);
                void (*forward_to)(const DeviceTensor*, void**, CallbackOnce*);
            };
            bool m_readonly;
            void* m_dev_nd;
            void* m_varptr;
            FuncTable* m_functable;
        public:
            uint32_t _version1;
    };

    /*!
     * \brief reinterpret_cast raw pointer or pointer integer to mgb object and
     *      check version
     * \return object pointer if the version is correct; nullptr if failed
     */
    template<typename T, typename S>
    T* as_versioned_obj(S &&val) {
        T *obj = reinterpret_cast<T*>(val);
        if (obj->_version0 != T::CURRENT_VERSION ||
                obj->_version1 != T::CURRENT_VERSION) {
            return nullptr;
        }
        return obj;
    }
} // namespace pubapi
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
