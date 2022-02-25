#pragma once

#include "megbrain/utils/thin/function.h"

namespace mgb {
namespace imperative {

template <typename TFunction>
class FunctionHook;

template <template <typename> class TFunction, typename TRet, typename... TArgs>
class FunctionHook<TFunction<TRet(TArgs...)>> {
public:
    using FunctionType = TFunction<TRet(TArgs...)>;
    explicit FunctionHook(FunctionType* fptr) : m_fptr{fptr} { m_backup = *fptr; }

public:
    template <
            typename THook,
            typename = std::enable_if_t<
                    std::is_invocable_r_v<TRet, THook, FunctionType, TArgs...>, void>>
    FunctionHook& apply_hook(THook&& hook) {
        // Replace with hooked version
        *m_fptr = [func = *m_fptr,
                   hook = std::forward<THook>(hook)](TArgs... args) -> TRet {
            return hook(func, std::forward<TArgs>(args)...);
        };
        // Convinent for chain call
        return *this;
    }

private:
    FunctionType* m_fptr;
    FunctionType m_backup;

public:
    ~FunctionHook() { *m_fptr = std::move(m_backup); }
};

template <typename TFunction>
auto make_shared_hook(TFunction* fptr) {
    return std::make_shared<FunctionHook<TFunction>>(fptr);
}

}  // namespace imperative
}  // namespace mgb
