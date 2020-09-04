/**
 * \file imperative/python/src/dispatcher.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./dispatcher.h"
#include "./pyext17.h"
#include "megbrain/exception.h"
#include "megbrain/utils/hash.h"
#include "megbrain/utils/small_vector.h"

#include <unordered_map>
#include <structmember.h>

namespace py = pybind11;
namespace pyx = pyext17;

namespace {

struct Handler {
    PyObject* func; // borrowed
    bool enabled;

    Handler() = default;
    Handler(PyObject* func_, bool enable = true) : func(func_), enabled(enable) {}
};

using FastSig = mgb::SmallVector<void*, 8>;
using MRO = std::vector<Handler*>;

struct Frame {
    MRO* mro;
    size_t mro_offset;

    Frame() = default;
    Frame(MRO* mro_, size_t mro_offset_ = 0) : mro(mro_), mro_offset(mro_offset_) {}
};

struct FastSigHash {
    size_t operator()(const FastSig& sig) const {
        auto* ptr = &sig.front();
        return mgb::XXHash()
            .update(ptr, sig.size() * sizeof(FastSig::value_type))
            .digest();
    }
};

struct ObjectIdHash : std::hash<void*> {
    size_t operator()(const py::handle& h) const {
        return std::hash<void*>::operator()(h.ptr());
    }
};

namespace {
using Container = std::vector<Frame>;
struct DispatcherStack: Container {
    constexpr static size_t MAX_RECURSIVE_DEPTH = 1024u;
    DispatcherStack() { reserve(MAX_RECURSIVE_DEPTH); }

    template<typename... Args>
    auto&& emplace_back_safely(Args&& ...args) {
        mgb_throw_if(size() >= MAX_RECURSIVE_DEPTH, mgb::MegBrainError,
            "recursion depth %zu is greater than the MAX_RECURSIVE_DEPTH(%zu)",
            size(), MAX_RECURSIVE_DEPTH);
        return emplace_back(std::forward<Args>(args)...);
    }
};
} // anonymous namespace

struct Dispatcher {
    std::unordered_map<FastSig, std::unique_ptr<MRO>, FastSigHash> cache;
    DispatcherStack stack;
    std::unordered_map<py::object, std::unique_ptr<Handler>, ObjectIdHash> registry;

    inline py::handle self() {
        return pyx::wrap<Dispatcher>::pycast(this);
    }

    bool prepare_call(PyObject*const* args, Py_ssize_t nargs) {
        FastSig sig(nargs);
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            sig[i] = Py_TYPE(args[i]);
        }
        auto it = cache.find(sig);
        if (it == cache.end()) {
            if (auto mro = resolve(sig)) {
                it = cache.emplace(std::move(sig), std::move(mro)).first;
            } else {
                return false;
            }
        }
        stack.emplace_back_safely(it->second.get());
        return true;
    }

    template<typename T>
    PyObject* do_call(T&& caller) {
        auto& frame = stack.back();
        auto& mro = *frame.mro;
        auto& i = frame.mro_offset;
        for (; i < mro.size(); ++i) {
            if (mro[i]->enabled) {
                auto ret = caller(mro[i]->func);
                if (ret != Py_NotImplemented) {
                    stack.pop_back();
                    return ret;
                }
                Py_DECREF(ret);
            }
        }
        PyErr_SetString(PyExc_NotImplementedError, "mro exhausted");
        stack.pop_back();
        return nullptr;
    }

    std::unique_ptr<MRO> resolve(const FastSig& sig) {
        try {
            py::tuple args(sig.size());
            for (size_t i = 0; i < sig.size(); ++i) {
                args[i] = (PyObject*)sig[i];
            }
            auto mro_iter = self().attr("dispatch_iter")(*args);
            auto ret = std::make_unique<MRO>();
            for (auto i : mro_iter) {
                auto it = registry.find(py::reinterpret_borrow<py::object>(i));
                if (it == registry.end()) {
                    PyErr_SetString(PyExc_RuntimeError, "resolved to unregistered function");
                    return nullptr;
                }
                ret->push_back(it->second.get());
            }
            return ret;
        } catch (py::error_already_set& e) {
            e.restore();
        } catch (std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return nullptr;
    }

public:
    static constexpr auto tp_name = "Dispatcher";

    PyObject* tp_vectorcall(PyObject*const* args, Py_ssize_t nargs) {
        if (!prepare_call(args, nargs)) return nullptr;
        return do_call([=](PyObject* func){return _PyObject_FastCall(func, const_cast<PyObject**>(args), nargs);});
    }

    PyObject* tp_call(PyObject* args, PyObject* kwargs) {
        if (!prepare_call(&PyTuple_GET_ITEM(args, 0), PyTuple_GET_SIZE(args))) return nullptr;
        return do_call([=](PyObject* func){return PyObject_Call(func, args, kwargs);});
    }

    PyObject* super(PyObject*const* args, Py_ssize_t nargs) {
        if (stack.empty()) {
            PyErr_SetString(PyExc_RuntimeError, "super called at top level");
            return nullptr;
        }
        stack.emplace_back_safely(stack.back()).mro_offset++;
        return do_call([=](PyObject* func){return _PyObject_FastCall(func, const_cast<PyObject**>(args), nargs);});
    }

    void enable(PyObject* func) {
        auto obj = py::reinterpret_borrow<py::object>(func);
        auto it = registry.find(obj);
        if (it != registry.end()) {
            it->second->enabled = true;
        } else {
            registry.emplace(std::move(obj), std::make_unique<Handler>(func));
        }
    }

    PyObject* disable(PyObject* func) {
        auto obj = py::reinterpret_borrow<py::object>(func);
        auto it = registry.find(obj);
        if (it == registry.end()) {
            PyErr_SetString(PyExc_ValueError, "function not registered");
            return nullptr;
        } else {
            it->second->enabled = false;
        }
        Py_RETURN_NONE;
    }

    void clear_cache() {
        cache.clear();
    }
};

} // namespace

void init_dispatcher(py::module m) {
    auto* dispatcher_type = pyx::wrap<Dispatcher>::type()
        .def<&Dispatcher::enable>("enable")
        .def<&Dispatcher::disable>("disable")
        .def<&Dispatcher::clear_cache>("clear_cache")
        .def<&Dispatcher::tp_vectorcall>("call")
        .def<&Dispatcher::super>("super")
        .finalize();
    if (!dispatcher_type) throw py::error_already_set();
    m.attr("Dispatcher") = dispatcher_type;
}
