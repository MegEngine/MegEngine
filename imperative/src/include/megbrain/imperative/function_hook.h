/**
 * \file imperative/src/include/megbrain/imperative/function_hook.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/thin/function.h"

namespace mgb {
namespace imperative {
template <typename TFunction>
class FunctionHooker;

template <typename TRet, typename... TArgs>
class FunctionHooker<TRet(TArgs...)> {
public:
    using FunctionType = thin_function<TRet(TArgs&&...)>;
    using HookType = thin_function<TRet(FunctionType, TArgs&&...)>;
    explicit FunctionHooker(FunctionType* fptr) : m_fptr{fptr} {}

public:
    FunctionHooker& apply_hook(HookType&& hook) {
        if (!m_backup) {
            FunctionType* backup = new FunctionType(*m_fptr);
            std::function<void(FunctionType*)> restorer =
                    [fptr = m_fptr](FunctionType* bkp) -> void {
                *fptr = *bkp;
                delete bkp;
            };
            m_backup = decltype(m_backup)(backup, restorer);
        }
        *m_fptr = [func = *m_fptr, hook](TArgs&&... args) -> TRet {
            return hook(func, std::forward<TArgs>(args)...);
        };
        return *this;
    }

private:
    FunctionType* m_fptr;
    std::unique_ptr<FunctionType, std::function<void(FunctionType*)>> m_backup;
};

template <typename TRet, typename... TArgs>
FunctionHooker(thin_function<TRet(TArgs...)>* f)
        ->FunctionHooker<TRet(TArgs...)>;
}  // namespace imperative

}  // namespace mgb
