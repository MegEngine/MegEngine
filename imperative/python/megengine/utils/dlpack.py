from ..core._imperative_rt.core2 import _from_dlpack

__all__ = [
    "to_dlpack",
    "from_dlpack",
]


def to_dlpack(tensor, stream=None):
    """
    Encodes a megengine tensor to DLPack.
    
    Args:
        tensor (Tensor): The input tensor, and the data type can be `float16`, `float32`,
                    `int8`, `int16`, `int32`, `uint8`, `uint16`, `complex64`.

        stream (Integer or None): An optional Python integer representing a CUDA stream,
                    The current stream is synchronized with this stream before the capsule is created.
                    If None or -1 is passed then no synchronization is performed.
    Returns:
        dltensor, and the data type is PyCapsule.

    Examples:
        .. code-block:: python

            import megengine as mge
            # x is a tensor with shape [2, 3]
            x = mge.tensor([[0.2, 0.3, 0.5],
                            [0.1, 0.2, 0.6]])
            dlpack = mge.utils.dlpack.to_dlpack(x)
            print(dlpack)
            # <capsule object "dltensor" at 0x7ff04b69cc00>
    
    """

    if stream is not None and stream != -1:
        return tensor.__dlpack__(stream)
    else:
        return tensor.__dlpack__()


def from_dlpack(ext_tensor, stream=None):
    """
    Decodes a DLPack to a megengine tensor.

    Args:
        ext_tensor (PyCapsule): a PyCapsule object with the dltensor.

        stream (Integer or None): An optional Python integer representing a CUDA stream,
                    user needs to know which stream the dlpack is generated.
                    If None then represent producers and consumers on the same stream.

    Returns:
        tensor (Tensor): a megengine tensor decoded from DLPack. 

    Examples:
        .. code-block:: python

            import megengine as mge
            # x is a tensor with shape [2, 3]
            x = mge.tensor([[0.2, 0.3, 0.5],
                            [0.1, 0.2, 0.6]])
            dlpack = mge.utils.dlpack.to_dlpack(x)
            x = mge.utils.dlpack.from_dlpack(dlpack)
            print(x)
            # Tensor([[0.2, 0.3, 0.5],
            #  [0.1, 0.2, 0.6]], device=gpu0:0) 
    """

    if isinstance(stream, int):
        assert stream >= 0, "device stream should be a positive value"
    stream = 0 if stream is None else stream
    return _from_dlpack(ext_tensor, stream)
