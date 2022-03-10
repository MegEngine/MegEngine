#pragma once

namespace mgb::imperative::python {

PyObject* dtype_promotion(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* get_device(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* make_shape_tuple(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* getitem_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* setitem_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* split_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* expand_dims_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* squeeze_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* transpose_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* broadcast_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* reshape_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* Const(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* astype_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* convert_single_value_cpp(PyObject* self, PyObject* const* args, size_t nargs);

PyObject* convert_inputs_cpp(PyObject* self, PyObject* const* args, size_t nargs);

}  // namespace mgb::imperative::python