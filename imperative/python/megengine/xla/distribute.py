import atexit
import warnings
from typing import Any, Optional, Sequence, Union

from .lib import xla_client as xc

xla_extention = xc._xla
xe = xla_extention


class State:
    process_id: int = 0
    ip: str = None
    port: int = None
    service: Optional[Any] = None
    client: Optional[Any] = None
    preemption_sync_manager: Optional[Any] = None
    visible_devices: Optional[str] = "all"

    def initialize(
        self,
        ip: str,
        port: int,
        num_processes: int,
        process_id: int,
        local_device_ids: Optional[Union[int, Sequence[int]]] = None,
    ):
        coordinator_address = ip + ":" + str(port)
        if local_device_ids is None:
            local_device_ids = [process_id]
        elif isinstance(local_device_ids, int):
            local_device_ids = [local_device_ids]
        else:
            local_device_ids = list(local_device_ids)

        assert local_device_ids == [process_id], f"{local_device_ids} .vs {process_id}"

        self.ip = ip
        self.port = port
        self.visible_devices = ",".join(str(x) for x in local_device_ids)
        self.process_id = process_id

        if process_id == 0:
            if self.service is not None:
                raise RuntimeError("distributed.initialize should only be called once.")
            self.service = xe.get_distributed_runtime_service(
                coordinator_address, num_processes, use_coordination_service=True
            )

        if self.client is not None:
            raise RuntimeError("distributed.initialize should only be called once.")

        # Set init_timeout to 5 min to leave time for all the processes to connect
        self.client = xe.get_distributed_runtime_client(
            coordinator_address,
            process_id,
            use_coordination_service=True,
            init_timeout=300,
        )
        self.client.connect()
        self.initialize_preemption_sync_manager()

    def shutdown(self):
        if self.client:
            self.client.shutdown()
            self.client = None
        if self.service:
            self.service.shutdown()
            self.service = None
        if self.preemption_sync_manager:
            self.preemption_sync_manager = None

    def initialize_preemption_sync_manager(self):
        if self.preemption_sync_manager is not None:
            raise RuntimeError(
                "Preemption sync manager should only be initialized once."
            )
        self.preemption_sync_manager = xe.create_preemption_sync_manager()
        self.preemption_sync_manager.initialize(self.client)


global_state = State()


def initialize(
    ip: str,
    port: int,
    num_processes: int,
    process_id: int,
    local_device_ids: Optional[Union[int, Sequence[int]]] = None,
):
    ip = "127.0.0.1" if ip == "localhost" else ip
    if global_state.service == None and global_state.client == None:
        global_state.initialize(ip, port, num_processes, process_id, local_device_ids)
        atexit.register(shutdown)
    else:
        assert (
            global_state.client != None
        ), "global_state.client should not be None if server is created"
        if global_state.ip == ip and global_state.port == port:
            return
        else:
            msg = (
                f"xla distribute server/client have been created on {global_state.ip}:{global_state.port}. "
                f"so ignore the request to create on {ip}:{port}"
            )
            warnings.warn(msg, category=RuntimeWarning)


def shutdown():
    global_state.shutdown()
