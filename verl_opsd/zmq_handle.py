from __future__ import annotations

import hashlib
import os
import tempfile


def build_zmq_ipc_handle(device_uuid: str) -> str:
    """Build a short, namespaced IPC socket handle for ZMQ weight transfer.

    The upstream default handle only keys on device UUID, which can collide when
    multiple runs share the same machine and GPU. We additionally include an
    optional run namespace and hash the result into a short filename so:

    1. different training runs do not fight over the same socket path
    2. the final AF_UNIX path stays comfortably below the 107-byte limit
    """

    namespace = os.getenv("VERL_ZMQ_NAMESPACE", "")
    unique_key = f"{namespace}:{device_uuid}" if namespace else device_uuid
    digest = hashlib.sha1(unique_key.encode("utf-8")).hexdigest()[:24]
    socket_path = os.path.join(tempfile.gettempdir(), f"rlz-{digest}.sock")
    return f"ipc://{socket_path}"
