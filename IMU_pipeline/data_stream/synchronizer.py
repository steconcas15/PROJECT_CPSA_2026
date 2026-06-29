# IMU_pipeline/synchronizer.py
# Collects Right IMU data (acc, gyr) and emits rows immediately to DataBuffer.
# Modified for a single BlueCoin device streaming (No dual-wrist requirements).

import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from utils.logger import log_system
from utils.config import get_bluecoin_config, get_sync_config
from IMU_pipeline.data_stream.data_buffer import DataBuffer

Vec3 = Tuple[float, float, float]

@dataclass
class _DevState:
    acc:  Optional[Vec3]  = None
    gyr:  Optional[Vec3]  = None
    ts_acc:  float = 0.0
    ts_gyr:  float = 0.0

    def ready(self) -> bool:
        return (self.acc is not None) and (self.gyr is not None)

    def clear(self) -> None:
        self.acc = self.gyr = None
        self.ts_acc = self.ts_gyr = 0.0


class IMUSynchronizer:
    """
    Receives single-wrist data from BLE listeners via update(device_id, kind, values, ts).
    Since only right wrist (bc_right) is alive, it forwards data directly to DataBuffer.
    """
    def __init__(self):
        self.right_id = "bc_right"

        log_system(f"[IMUSync] Monochannel initialization for right device: {self.right_id}")

        self._lock = threading.Lock()
        self._state: Dict[str, _DevState] = {
            self.right_id: _DevState()
        }
        self.buffer = DataBuffer()
        self._emits = 0
        self._pending_row = None

    def update(self, device_id: str, kind: str, values, ts: float) -> None:
        """
        device_id must match 'bc_right'.
        kind: {'acc','gyr'}.
        """
        if device_id != self.right_id:
            return

        row = None
        with self._lock:
            st = self._state[device_id]
            try:
                if kind == "acc":
                    st.acc = (float(values[0]), float(values[1]), float(values[2]))
                    st.ts_acc = ts
                elif kind == "gyr":
                    st.gyr = (float(values[0]), float(values[1]), float(values[2]))
                    st.ts_gyr = ts
                else:
                    return
            except Exception as e:
                log_system(f"[IMUSync] Bad packet extraction for {kind}: {e}", level="WARNING")
                return

            self._try_emit_locked()
            row = self._pending_row
            self._pending_row = None

        if row:
            (Racc, Rgyr, ts_emit) = row
            self.buffer.add_buffer_row(Racc, Rgyr, ts_emit)

    def _try_emit_locked(self) -> None:
        st = self._state[self.right_id]
        if st.ready():
            ts_emit = max(st.ts_acc, st.ts_gyr)
            row = (st.acc, st.gyr, ts_emit)
            self._emits += 1
            st.clear()
            self._pending_row = row

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {"emits": self._emits, "drops_left": 0, "drops_right": 0}

    def reset(self) -> None:
        with self._lock:
            self._state[self.right_id].clear()
            self._emits = 0