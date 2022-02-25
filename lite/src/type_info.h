#pragma once

#include "misc.h"

namespace lite {
/*!
 * \brief an object to represent a type
 *
 * LITE has a lightweight RTTI system. Each type is represented by the
 * address of a Typeinfo object, which is stored in the .bss segment.
 *
 * LITE_TYPEINFO_OBJ_DECL should be placed into the definition of classes that
 * need compile-time type support.
 *
 * For classes that need RTTI, they should be derived from DynTypeObj
 */
struct Typeinfo {
    //! name of the corresponding type; nullptr if MGB_VERBOSE_TYPEINFO_NAME==0
    const char* const name;

    /*!
     * \brief whether this is the type of given object
     * \tparam T a class with static typeinfo() method
     */
    template <typename T>
    bool is() const {
        return T::typeinfo() == this;
    }
};

/*!
 * \brief base class to emulate RTTI without compiler support
 */
class DynTypeObj {
public:
    virtual Typeinfo* dyn_typeinfo() const = 0;

    //! cast this to a final object with type check
    template <class T>
    T& cast_final_safe() {
        LITE_ASSERT(
                T::typeinfo() == dyn_typeinfo(), "can not convert type %s to %s",
                dyn_typeinfo()->name, T::typeinfo()->name);
        return *static_cast<T*>(this);
    }

    template <class T>
    const T& cast_final_safe() const {
        return const_cast<DynTypeObj*>(this)->cast_final_safe<T>();
    }

    //! check whether this is same to given type
    template <class T>
    bool same_type() const {
        return dyn_typeinfo() == T::typeinfo();
    }

protected:
    ~DynTypeObj() = default;
};

//! put in the declaration of a final class inherited from DynTypeObj
#define LITE_DYN_TYPE_OBJ_FINAL_DECL                                    \
public:                                                                 \
    ::lite::Typeinfo* dyn_typeinfo() const override final;              \
    static inline ::lite::Typeinfo* typeinfo() { return &sm_typeinfo; } \
                                                                        \
private:                                                                \
    static ::lite::Typeinfo sm_typeinfo

#if LITE_ENABLE_LOGGING
//! get class name from class object
#define _LITE_TYPEINFO_CLASS_NAME(_cls) #_cls
#else
#define _LITE_TYPEINFO_CLASS_NAME(_cls) nullptr
#endif

//! put in the impl file of a class that needs static typeinfo()
#define LITE_TYPEINFO_OBJ_IMPL(_cls) \
    ::lite::Typeinfo _cls::sm_typeinfo { _LITE_TYPEINFO_CLASS_NAME(_cls) }

//! put in the impl file of a final class inherited from DynTypeObj
#define LITE_DYN_TYPE_OBJ_FINAL_IMPL(_cls)                                \
    ::lite::Typeinfo* _cls::dyn_typeinfo() const { return &sm_typeinfo; } \
    LITE_TYPEINFO_OBJ_IMPL(_cls)

}  // namespace lite
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
