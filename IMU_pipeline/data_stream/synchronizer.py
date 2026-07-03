# imu_pipeline/data_stream/synchronizer.py
# Collects asynchronous data from a single Bluecoin device (accelerometer and gyroscope) 
# and emits atomically aggregated 6-axis rows immediately to the data buffer.

import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from utils.logger import log_system
from IMU_pipeline.data_stream.data_buffer import DataBuffer

# Define a type alias for 3d vector data (x, y, z coordinates)
Vec3 = Tuple[float, float, float]

@dataclass
class DeviceState:
    """
    Temporary data structure that stores incoming sensor samples until 
    a complete 6-axis frame can be assembled.
    """
    acc: Optional[Vec3] = None
    gyr: Optional[Vec3] = None
    ts_acc: float = 0.0
    ts_gyr: float = 0.0

    def is_ready(self) -> bool:
        """
        Evaluates aggregation readiness.
        
        Returns true only if both the asynchronous accelerometer and gyroscope 
        samples have successfully arrived for the current time frame.
        """
        return (self.acc is not None) and (self.gyr is not None)

    def clear(self) -> None:
        """Resets the state container to prepare for the next asynchronous data cycle."""
        self.acc = None
        self.gyr = None
        self.ts_acc = 0.0
        self.ts_gyr = 0.0


class IMUSynchronizer:
    """
    Receives raw multi-sensor data from a single Bluecoin device via bluetooth callbacks.
    
    This class handles the asynchronous nature of independent bluetooth characteristic 
    notifications. It aggregates separate accelerometer and gyroscope updates into a 
    complete 6-axis frame and guarantees thread-safe atomicity during state changes 
    using a mutual exclusion lock.
    """
    def __init__(self):
        self.device_id = "bc_left"
        log_system(f"[IMUSync] Initialized for single-device channel: {self.device_id}")

        # The threading lock ensures atomicity. Because bluetooth notifications execute on 
        # separate background threads, this lock prevents race conditions, ensuring 
        # that no partial or corrupted data is read or emitted mid-update.
        self._lock = threading.Lock()
        self._state = DeviceState()
        self.buffer = DataBuffer()
        self._emits = 0

    def update(self, device_id: str, kind: str, values, ts: float) -> None:
        """
        Processes incoming data notifications from background bluetooth threads.
        
        The method updates the specific sensor slot ('acc' or 'gyr') as packets arrive 
        independently at unpredictable millisecond intervals. To ensure consistency, 
        state modifications are wrapped inside a critical section using a lock. Once 
        both slots are filled, it aggregates the data into a single completed 6-axis row 
        and pushes it forward to the data buffer.
        """
        if device_id != self.device_id:
            return

        row_to_emit = None
        
        # Acquire the lock to ensure all-or-nothing state updates and evaluations.
        with self._lock:
            try:
                if kind == "acc":
                    self._state.acc = (float(values[0]), float(values[1]), float(values[2]))
                    self._state.ts_acc = ts
                elif kind == "gyr":
                    self._state.gyr = (float(values[0]), float(values[1]), float(values[2]))
                    self._state.ts_gyr = ts
                else:
                    return
            except Exception as e:
                log_system(f"[IMUSync] Failed to extract packet data for {kind}: {e}", level="WARNING")
                return

            # Aggregation check: verify if the complete 6-axis payload is ready.
            # If both asynchronous components are present, combine them immediately.
            if self._state.is_ready():
                # Align timestamps by selecting the most recent emission time
                ts_emit = max(self._state.ts_acc, self._state.ts_gyr)
                row_to_emit = (self._state.acc, self._state.gyr, ts_emit)
                self._emits += 1
                
                # Reset the temporary state container as part of the atomic transaction
                self._state.clear()

        # Forwarding data to the data buffer is executed outside the lock.
        # This prevents blocking background bluetooth notification threads during input/output operations.
        if row_to_emit:
            acc_vals, gyr_vals, ts_emit = row_to_emit
            self.buffer.add_buffer_row(acc_vals, gyr_vals, ts_emit)

    def get_stats(self) -> Dict[str, int]:
        """Thread-safe retrieval of emission statistics."""
        with self._lock:
            return {"emits": self._emits}

    def reset(self) -> None:
        """Thread-safe reset of the internal synchronization state."""
        with self._lock:
            self._state.clear()
            self._emits = 0
