# sharding annotation helper, but we do not use the sharding in megengine
# so we set all the input sharding as `Replicated` by default
import abc
import functools
import itertools as it
from typing import Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..tensor import Parameter as MgeParameter
from ..tensor import Tensor as MgeTensor
from .device import device_put
from .dtype import _np_types, canonicalize_arg
from .lib import xla_client as xc
from .utils import safe_zip, tuple_insert, unzip3, use_cpp_class, use_cpp_method

pmap_lib = xc._xla.pmap_lib


def spec_to_indices(shape, spec):
    shape = (1, *shape)
    return tuple(spec.indices(shape).flat)


@use_cpp_class(xc.Sharding)
class Sharding(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def device_set(self):
        raise NotImplementedError("should be overrided")

    @abc.abstractmethod
    def devices_indices_map(self, global_shape):
        raise NotImplementedError("should be overrided")

    @abc.abstractmethod
    def shard_shape(self, global_shape):
        raise NotImplementedError("should be overrided")

    @abc.abstractmethod
    def is_equivalent_to(self, other, ndim) -> bool:
        raise NotImplementedError("should be overrided")

    @functools.cached_property
    def addressable_devices(self):
        return {
            d for d in self.device_set if d.process_index == d.client.process_index()
        }

    @functools.cached_property
    def is_fully_addressable(self) -> bool:
        return len(self.device_set) == len(self.addressable_devices)

    @functools.lru_cache(maxsize=4096)
    def addressable_devices_indices_map(self, global_shape):
        return {
            d: ind
            for d, ind in self.devices_indices_map(global_shape).items()
            if d.process_index == d.client.process_index()
        }


@use_cpp_class(xc.XLACompatibleSharding)
class XLACompatibleSharding(Sharding, metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def _device_assignment(self):
        raise NotImplementedError("should be overrided")

    @abc.abstractmethod
    def _to_xla_op_sharding(self, num_dimensions: int):
        raise NotImplementedError("should be overrided")

    @functools.lru_cache(maxsize=4096)
    def devices_indices_map(self, global_shape):
        op_sharding = self._to_xla_op_sharding(len(global_shape))
        op_sharding_sharding = OpShardingSharding(self._device_assignment, op_sharding)
        return op_sharding_sharding.devices_indices_map(global_shape)

    @functools.cached_property
    def _addressable_device_assignment(self):
        return [
            d
            for d in self._device_assignment
            if d.process_index == d.client.process_index()
        ]


@use_cpp_class(xc.GSPMDSharding)
class OpShardingSharding(XLACompatibleSharding):
    @use_cpp_method
    def __init__(self, devices, op_sharding):
        self._devices = tuple(devices)
        self._op_sharding = op_sharding

    def __reduce__(self):
        return type(self), (self._devices, self._op_sharding)

    @functools.cached_property
    def _op_sharding_hash(self):
        return hash(xc.HloSharding.from_proto(self._op_sharding))

    def __eq__(self, other):
        if not isinstance(other, OpShardingSharding):
            return False
        if id(self) == id(other):
            return True

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((self._devices, self._op_sharding_hash))
        return self._hash

    def __repr__(self):
        return (
            f"OpShardingSharding({repr(xc.HloSharding.from_proto(self._op_sharding))})"
        )

    @functools.cached_property
    def device_set(self):
        return set(self._devices)

    @property
    def _device_assignment(self):
        return list(self._devices)

    def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
        return self._op_sharding

    @classmethod
    def get_replicated(cls, device_assignment):
        proto = _get_replicated_op_sharding()
        return cls(device_assignment, proto)


@use_cpp_class(xc.SingleDeviceSharding)
class SingleDeviceSharding(XLACompatibleSharding):
    @use_cpp_method
    def __init__(self, device):
        self._device = device

    def __reduce__(self):
        return type(self), (self._device,)

    def __repr__(self):
        return f"SingleDeviceSharding(device={repr(self._device)})"

    def __hash__(self):
        return hash(self._device)

    def __eq__(self, other):
        if not isinstance(other, SingleDeviceSharding):
            return False
        if id(self) == id(other):
            return True
        return self._device == other._device

    @property
    def device_set(self):
        return {self._device}

    def devices_indices_map(self, global_shape):
        return {self._device: (slice(None),) * len(global_shape)}

    @property
    def _device_assignment(self):
        return [self._device]

    def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
        return _get_replicated_op_sharding()


@use_cpp_class(xc.PmapSharding)
class PmapSharding(XLACompatibleSharding):
    devices: np.ndarray
    sharding_spec: pmap_lib.ShardingSpec

    @use_cpp_method
    def __init__(
        self,
        devices: Union[Sequence[xc.Device], np.ndarray],
        sharding_spec: pmap_lib.ShardingSpec,
    ):
        self.devices = np.asarray(devices)
        self.sharding_spec = sharding_spec

    def __reduce__(self):
        return type(self), (self.devices, self.sharding_spec)

    def __eq__(self, other):
        if not isinstance(other, PmapSharding):
            return False
        if id(self) == id(other):
            return True
        return self.sharding_spec == other.sharding_spec and np.array_equal(
            self.devices, other.devices
        )

    def __hash__(self):
        if not hasattr(self, "_hash"):
            self._hash = hash((tuple(self.devices.flat), self.sharding_spec))
        return self._hash

    def __str__(self):
        device_ids = [d.id for d in self.devices.flat]
        return (
            f"PmapSharding(sharding_spec={self.sharding_spec}, "
            f"{device_ids=}, "
            f"device_platform={self.devices.flat[0].platform.upper()}, "
            f"device_shape={self.devices.shape})"
        )

    def __repr__(self):
        return (
            f"PmapSharding(sharding_spec={self.sharding_spec}, "
            f"devices={self.devices})"
        )

    def is_equivalent_to(self, other, ndim: int,) -> bool:
        return self == other

    @functools.cached_property
    def device_set(self) -> Set[xc.Device]:
        return set(self.devices.flat)

    @functools.lru_cache(maxsize=4096)
    def devices_indices_map(self, global_shape):
        indices = spec_to_indices(global_shape, self.sharding_spec)
        return dict(safe_zip(self.devices.flat, indices))

    @functools.cached_property
    def _device_assignment(self):
        return list(self.devices.flat)

    def _to_xla_op_sharding(self, num_dimensions: int) -> xc.OpSharding:
        raise NotImplementedError("pmap doesn't use OpSharding.")

    @functools.lru_cache(maxsize=4096)
    def shard_shape(self, global_shape):
        sharded_dim = None
        sharded_dim_size = None
        for i, s in enumerate(self.sharding_spec.sharding):
            if isinstance(s, pmap_lib.Unstacked):
                sharded_dim = i
                sharded_dim_size = s.size
                break
        if sharded_dim is None:
            return global_shape
        if global_shape[sharded_dim] != sharded_dim_size:
            raise ValueError(
                f"The sharded dimension must be equal to the number of "
                f"devices passed to PmapSharding. Got sharded dimension {sharded_dim} "
                f"with value {global_shape[sharded_dim]} in shape {global_shape} and "
                f"the number of devices={len(self._device_assignment)}"
            )
        return global_shape[:sharded_dim] + global_shape[sharded_dim + 1 :]


def _get_op_sharding_shardings_from_executable(
    xla_executable,
    device_assignment: Sequence[xc.Device],
    num_in_avals: int,
    num_out_avals: int,
) -> Tuple[
    Sequence[XLACompatibleSharding], Sequence[XLACompatibleSharding],
]:
    assert len(device_assignment) == 1
    if len(device_assignment) == 1:
        return (
            [SingleDeviceSharding(device_assignment[0]) for _ in range(num_in_avals)],
            [SingleDeviceSharding(device_assignment[0]) for _ in range(num_out_avals)],
        )


def is_op_sharding_replicated(op: xc.OpSharding) -> bool:
    if len(op.tile_assignment_devices) == 1:
        return True
    return xc.HloSharding.from_proto(op).is_replicated()


class _UnspecifiedSharding:
    pass


def _is_unspecified(x):
    return isinstance(x, _UnspecifiedSharding)


def make_unspec_sharding(inps):
    return [_UnspecifiedSharding()] * len(inps)


# split the tensor into sharded shape according to the sharding strategy
def sharded_val(in_val, in_sharding):
    if in_sharding is None or _is_unspecified(in_sharding):
        return in_val

    if in_sharding.type == xc.OpSharding.Type.REPLICATED:
        return in_val

    assert False, "not implemented"


def _get_normalized_avals_and_shardings(
    global_in_avals, in_shardings, in_is_global,
):
    avals = []
    shardings = []

    for gaval, i, is_global in safe_zip(global_in_avals, in_shardings, in_is_global):
        if is_global:
            aval = gaval
            in_sharding = i
        else:
            assert False
        avals.append(aval)
        shardings.append(in_sharding)

    return avals, shardings


shard_arg_handlers = {}


def _shard_nparray(x, devices, indices, sharding=None):
    if x.shape == ():
        return device_put([x] * len(devices), devices)
    return device_put([x[i] for i in indices], devices)


def _shard_xla_device_array(x: xc._xla.DeviceArray, devices, indices, sharding=None):
    def _as_slice_indices(arr, idx):
        start_indices = [0] * arr.ndim
        limit_indices = list(arr.shape)
        removed_dims = []

        tuple_idx = idx if isinstance(idx, tuple) else (idx,)
        for dim, sub_idx in enumerate(tuple_idx):
            if isinstance(sub_idx, int):
                start_indices[dim] = sub_idx
                limit_indices[dim] = sub_idx + 1
                removed_dims.append(dim)
            elif sub_idx == slice(None):
                continue
            else:
                assert isinstance(sub_idx, slice), sub_idx
                assert isinstance(sub_idx.start, int), sub_idx
                assert isinstance(sub_idx.stop, int), sub_idx
                start_indices[dim] = sub_idx.start
                limit_indices[dim] = sub_idx.stop

        return tuple(start_indices), tuple(limit_indices), tuple(removed_dims)

    start_indices, limit_indices, removed_dims = unzip3(
        _as_slice_indices(x, idx) for idx in indices
    )
    shards = x._multi_slice(start_indices, limit_indices, removed_dims)
    return device_put(shards, devices)


def _shard_mge_tensor(x, devices, indices, sharding=None):
    x_np = x.numpy()
    if x_np.shape == ():
        x_np = np.array([x_np])
    return device_put([x_np[i] for i in indices], devices)


for nt in _np_types:
    shard_arg_handlers[nt] = _shard_nparray
shard_arg_handlers[xc._xla.DeviceArray] = _shard_xla_device_array
shard_arg_handlers[MgeTensor] = _shard_mge_tensor
shard_arg_handlers[MgeParameter] = _shard_mge_tensor


def shard_args(devices, indices, args, shardings=None):
    def _shard_arg(arg, devices, arg_indices, sharding=None):
        arg = canonicalize_arg(arg)
        return shard_arg_handlers[type(arg)](arg, devices, arg_indices, sharding)

    if shardings is None:
        return [_shard_arg(arg, devices, indices[i]) for i, arg in enumerate(args)]
    else:
        return [
            _shard_arg(arg, devices, indices[i], shardings[i])
            for i, arg in enumerate(args)
        ]


@functools.lru_cache()
def _get_replicated_op_sharding():
    proto = xc.OpSharding()
    proto.type = xc.OpSharding.Type.REPLICATED
    return proto


def partitioned_sharding_spec(
    num_partitions: int, partitions: Optional[Sequence[int]], arg_shape
):
    if partitions is None:
        maybe_replicate = (
            () if num_partitions == 1 else (pmap_lib.Replicated(num_partitions),)
        )
        return pmap_lib.ShardingSpec(
            sharding=[pmap_lib.NoSharding()] * len(arg_shape),
            mesh_mapping=maybe_replicate,
        )
    else:
        assert len(partitions) == len(arg_shape)
        return pmap_lib.ShardingSpec(
            sharding=map(pmap_lib.Chunked, [[x] for x in partitions]),
            mesh_mapping=map(pmap_lib.ShardedAxis, range(len(partitions))),
        )


def _pmap_sharding_spec(
    nrep, axis_size, npart, parts, arg_shape, map_axis: Optional[int]
) -> pmap_lib.ShardingSpec:
    replication_factor, ragged = divmod(nrep, axis_size)
    assert not ragged
    # get the sharding spec from inner sharded_jits as if we weren't in a pmap
    pspec = partitioned_sharding_spec(npart, parts, arg_shape)
    maybe_replicate = (
        () if replication_factor == 1 else (pmap_lib.Replicated(replication_factor),)
    )
    if map_axis is not None:
        sharded_in_axis = sum(
            not isinstance(s, pmap_lib.NoSharding) for s in pspec.sharding[:map_axis]
        )

        def shift_sharded_axis(a):
            if isinstance(a, pmap_lib.ShardedAxis) and a.axis >= sharded_in_axis:
                return pmap_lib.ShardedAxis(a.axis + 1)
            return a

        # replication_factor represents the product of inner pmaps, so it goes
        # after the outer pmapped axis at index 0
        return pmap_lib.ShardingSpec(
            sharding=tuple_insert(
                pspec.sharding, map_axis, pmap_lib.Unstacked(axis_size)
            ),
            mesh_mapping=it.chain(
                [pmap_lib.ShardedAxis(sharded_in_axis)],
                maybe_replicate,
                map(shift_sharded_axis, pspec.mesh_mapping),
            ),
        )
    else:
        return pmap_lib.ShardingSpec(
            sharding=pspec.sharding,
            mesh_mapping=(pmap_lib.Replicated(axis_size),)
            + maybe_replicate
            + pspec.mesh_mapping,
        )


def _get_pmap_sharding(devices, specs):
    return [PmapSharding(devices, spec) for spec in specs]
