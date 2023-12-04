#pragma once
#include <cstddef>
#include <functional>
#include <thread>

#if !defined(__APPLE__)
#define USE_STL_THREAD_LOCAL 1
#else
#define USE_STL_THREAD_LOCAL 0
#endif

// clang-format off
#if defined(__APPLE__)
#   if (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ + 0) >= 101000
#       define USE_STL_THREAD_LOCAL 1
#   else
#       define USE_STL_THREAD_LOCAL 0
#   endif
#endif

#if defined(__ANDROID__) || defined(__OHOS__)
#pragma message("force disable USE_STL_THREAD_LOCAL for thread_local mem leak at dlopen/dlclose")
#undef USE_STL_THREAD_LOCAL
#define USE_STL_THREAD_LOCAL 0
class ThreadData;
class ThreadLocalForceFree {
public:
    ThreadLocalForceFree() = default;
    ~ThreadLocalForceFree() {
        MGB_LOCK_GUARD(m_mutex);
        for (auto& d : td) {
            delete (ThreadData*)d;
        }
    }

    void push(void* d);

private:
    std::vector<void*> td;
    MGB_MUTEX m_mutex;
};
ThreadLocalForceFree& get_thread_local_force_free_instance();
#endif

#if USE_STL_THREAD_LOCAL
#define MGB_THREAD_LOCAL_PTR(T) thread_local T*
#else
#define MGB_THREAD_LOCAL_PTR(T) ThreadLocalPtr<T>
#endif
// clang-format on

#if !USE_STL_THREAD_LOCAL
#include <pthread.h>

namespace mgb {

template <typename T>
class ThreadLocalPtr {
    struct ThreadData {
        const ThreadLocalPtr* self = nullptr;
        T** data = nullptr;
    };
    pthread_key_t m_key;
    std::function<T**()> m_constructor = nullptr;
    std::function<void(T**)> m_destructor = nullptr;

    void move_to(T** data) {
        if (void* d = pthread_getspecific(m_key)) {
            *data = *static_cast<ThreadData*>(d)->data;
        }
    }

    T** get() const {
        if (auto d = pthread_getspecific(m_key)) {
            return static_cast<ThreadData*>(d)->data;
        }
        ThreadData* t_data = new ThreadData();
#if defined(__ANDROID__) || defined(__OHOS__)
        get_thread_local_force_free_instance().push((void*)t_data);
#endif

        t_data->data = m_constructor();
        t_data->self = this;
        pthread_setspecific(m_key, t_data);
        return t_data->data;
    }

    static void exit(void* d) {
        ThreadData* td = static_cast<ThreadData*>(d);
        if (td && td->self->m_destructor)
            td->self->m_destructor(td->data);
#if !defined(__ANDROID__) && !defined(__OHOS__)
        delete td;
#endif
    }

public:
    ThreadLocalPtr(
            std::function<T**()> constructor,
            std::function<void(T**)> destructor = std::default_delete<T*>())
            : m_constructor(constructor), m_destructor(destructor) {
        pthread_key_create(&m_key, exit);
    }

    ThreadLocalPtr() : ThreadLocalPtr(std::function<T**()>([] { return new T*(); })) {}

    ThreadLocalPtr(std::nullptr_t) : ThreadLocalPtr([] { return new T*(nullptr); }) {}

    ThreadLocalPtr(ThreadLocalPtr&& other) : ThreadLocalPtr() { other.move_to(get()); }

    ThreadLocalPtr& operator=(ThreadLocalPtr&& other) {
        other.move_to(get());
        return *this;
    }
    ThreadLocalPtr& operator=(T* v) {
        *get() = v;
        return *this;
    }
    ~ThreadLocalPtr() { pthread_key_delete(m_key); }

    //!& operator like std thread_local
    T** operator&() const { return get(); }

    //! use in if()
    operator bool() const { return *get(); }

    //! directly access its member
    T* operator->() const { return *get(); }

    //! convert to T*
    operator T*() const { return *get(); }
};

}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
