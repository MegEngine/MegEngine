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
    using FunctionType = thin_function<TRet(TArgs...)>;
    //Type of hooks. Hook should accept a real function as argument
    //and invoke it on an appropriate time
    using HookType = thin_function<TRet(FunctionType, TArgs...)>;
    explicit FunctionHooker(FunctionType* fptr) : m_fptr{fptr} {
        m_backup = {nullptr, [](FunctionType*){}};
    }

public:
    FunctionHooker& apply_hook(HookType&& hook) {
        if (!m_backup) {
            FunctionType* backup = new FunctionType(*m_fptr);
            //Restore hooked function, would be invoked when destructed
            std::function<void(FunctionType*)> restorer =
                    [fptr = m_fptr](FunctionType* bkp) -> void {
                *fptr = *bkp;
                delete bkp;
            };
            m_backup = decltype(m_backup)(backup, restorer);
        }
        //Replace with hooked version
        *m_fptr = [func = *m_fptr, hook](TArgs... args) -> TRet {
            return hook(func, std::forward<TArgs>(args)...);
        };
        //Convinent for chain call
        return *this;
    }

private:
    FunctionType* m_fptr;
    std::unique_ptr<FunctionType, std::function<void(FunctionType*)>> m_backup;
};

//Helps to deduce template args
template <typename TRet, typename... TArgs>
FunctionHooker(thin_function<TRet(TArgs...)>* f)
        -> FunctionHooker<TRet(TArgs...)>;

template<typename TSignature>
auto make_shared_hook(thin_function<TSignature>* fptr){
    return std::make_shared<FunctionHooker<TSignature>>(fptr);
}

}  // namespace imperative
}  // namespace mgb
