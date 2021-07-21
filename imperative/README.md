## Build

Same as [MegEngine](/README.md) except passing the additional flag "-DMGE_BUILD_IMPERATIVE_RT=ON" to cmake configure command.

## Test

1. Make sure `make develop` is executed

2. Setup `PYTHONPATH`

   ```bash
   export PYTHONPATH="$(git rev-parse --show-toplevel)/imperative/python"
   ```
3. Run `pytest` (pip install as needed)

   ```bash
   cd $(git rev-parse --show-toplevel)/imperative/python/test
   pytest
   ```

## Concepts

### Op and Tensor-like

An op is a subclass of `OpBase` representing some operation, for example

* `Elemwise`
* `Reduce`

Op can be parametrized. For example, `Elemwise` has a single parameter `mode`, which is required by its constructor.

A tensor-like is a subclass of `TensorBase` that defines how ops should apply on it, for example

* `RawTensor` launch kernel associated with op
* `Tracer` record information for autodiff

Op instances are callable with signature `(*args: TensorBase) -> Tuple[TensorBase]`. It will invoke the correct implementation for that specific op and tensor-like, e.g. launch kernel if `args` is `RawTensor`, record information for autodiff if `args` is `Tracer`.

### The `Const` Op

The `Const` op is a special op that is used to convert literal to tensor-likes. Although it does not really use any input, at least one should be provided, otherwise it can't know which specific tensor-like to return.

### Tensor Wrapper

Tensor-likes have a dataflow semantic, thus immutable. `TensorWrapper` provide a mutable layer on top of tensor-likes by replacing the wrapped tensor-like on demand.

## How to Wrap a MegBrain Op

1. Define the op

   Most ops have been automatically generated in `ops.builtin` using `.oprdecl` files (take a look at [basic_arith.oprdecl](/src/opr/impl/basic_arith.oprdecl)). If your op is already there, skip to next step.

   For other ops, this work can still be done automatically with the help of an Python op serializer that matches MegBrain's own.

   Before proceeding, if you are unfamiliar with MegBrain's serializer, here is a brief introduction. Each MegBrain op has a registered name, which is found at `MGB_SEREG_OPR(this_is_the_name, ...)` in some `.sereg.h` file. The default serializer simply write the memory of struct returned by `opr->param()`.

   You can create a serializer by subclassing `ops._internal.helper.OpDef` as follows

   ```python
   class WhateverDef(OpDef): # must end with "Def"
       name = 'Whatever' # name in MegBrain serialization registry
       param_names = ('param',) # Does not have to be 'param', but it is a good practice to mirror
                                # C++ name, which is usually param(). It can also contain more
                                # than one element, for example if the C++ serializer writes
                                # `opr->param1()` followed by `opr->param2()`, you should use
                                # ('param1', 'param2') instead.

       class Param:
           def serialize(self):
               c_struct_memory = bytes(...) # memory of a C++ `Param` struct
               return b'\x00'*4 + c_struct_memory # remember to add 4 leading bytes

       def __init__(self):
           self.param = self.Param(...) # must assign to attribute(s) specified in param_names
   ```

   A concrete example can be found at `ops._internal.misc_ops.DimshuffleDef`.

   Lastly, make sure it is imported in `ops._internal.all_ops` and a corresponding op will show up in `ops.builtin`

2. Define a convenience function

   Use `functional` as a reference.

   Tips:

   * an op instance has to be constructed before applying it

      `op = WhateverOp(param=...)`

   * apply an op by calling the op instance

      `outputs = op(*inputs)`

   * op always return a tuple

      `result, = outputs`

   * input can be any tensor-like
