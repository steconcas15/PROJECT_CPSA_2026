# feature_listeners.py
# Feature listeners modules for BlueCoin sensor data callbacks (accelerometer, gyroscope)

from blue_st_sdk.feature import FeatureListener
from utils.logger import log_system
import time
from typing import Sequence, Optional, Tuple


def _to_floats(data: Sequence, n: int) -> Optional[Tuple[float, ...]]:
    """ Convert the first n items of data to floats """
    if not data or len(data) < n:
        return None
    try:
        return tuple(float(data[i]) for i in range(n))
    except Exception:
        return None


class AccelerometerFeatureListener(FeatureListener):
    """ Listens for accelerometer data updates from BlueCoin right device. """
    def __init__(self, device_id: str, synchronizer):
        super().__init__()
        self.device_id = device_id
        self.sync = synchronizer

    def on_update(self, feature, sample):
        try:
            values = _to_floats(sample.get_data(), 3)
            if not values:
                return
            self.sync.update(self.device_id, "acc", values, ts=time.monotonic())
        except Exception as e:
            log_system(f"[Accelerometer Listener: {self.device_id}] {type(e).__name__}: {e}", level="ERROR")


class GyroscopeFeatureListener(FeatureListener):
    """ Listens for gyroscope data updates from BlueCoin right device. """
    def __init__(self, device_id: str, synchronizer):
        super().__init__()
        self.device_id = device_id
        self.sync = synchronizer

    def on_update(self, feature, sample):
        try:
            values = _to_floats(sample.get_data(), 3)
            if not values:
                return
            self.sync.update(self.device_id, "gyr", values, ts=time.monotonic())
        except Exception as e:
            log_system(f"[Gyroscope Listener: {self.device_id}] {type(e).__name__}: {e}", level="ERROR")