# IMU_pipeline/data_buffer.py
# Sliding window buffer for a single BlueCoin (Right wrist only, Accel + Gyro).

from __future__ import annotations

from typing import List, Tuple, Callable, Optional
import threading
import numpy as np

from utils.logger import log_system
from utils.config import get_buffer_config

Vec3 = Tuple[float, float, float]

class DataBuffer:
    """
    Collects aligned rows from single device synchronizer and generates window contexts.
    Passes data directly to the DrowsinessClassifier pipeline.
    """
    def __init__(self, window_size: Optional[int] = None, hop_size: Optional[int] = None):
        cfg = get_buffer_config() or {}

        self.window_size = int(window_size if window_size is not None else cfg.get("window_size", 150))
        self.hop_size    = int(hop_size    if hop_size    is not None else cfg.get("overlap", 75))
        self._debug_print_buffer = bool(cfg.get("debug_print_buffer", False))

        self._rows: List[Tuple[float, ...]] = []   # ciascuna riga ha 6 float
        self._ts:   List[float] = []               # timestamp associati paralleli
        self._lock = threading.Lock()

        self._features_sink: Optional[Callable[[dict, float], None]] = None
        self._windows_emitted = 0
        self._calibrated = False

        log_system(f"[DataBuffer] Pipeline Ready: window= {self.window_size} hop= {self.hop_size}")

    def add_buffer_row(self, R_acc: Vec3, R_gyr: Vec3, ts_emit: float) -> None:
        """ Append one 6-axis sample to storage. """
        window_rows = None
        window_ts = None
        row = (
            float(R_acc[0]), float(R_acc[1]), float(R_acc[2]),  # RIGHT acc_x, acc_y, acc_z
            float(R_gyr[0]), float(R_gyr[1]), float(R_gyr[2]),  # RIGHT gyr_x, gyr_y, gyr_z
        )

        with self._lock:
            self._rows.append(row)
            self._ts.append(float(ts_emit))

            if len(self._rows) >= self.window_size:
                start = len(self._rows) - self.window_size
                window_rows = self._rows[start : start + self.window_size]
                window_ts   = self._ts[start   : start + self.window_size]

                # Slide forward
                keep_from = self.hop_size
                if keep_from <= 0:
                    self._rows.clear()
                    self._ts.clear()
                else:
                    self._rows = self._rows[keep_from:]
                    self._ts   = self._ts[keep_from:]

        if window_rows is not None:
            self._on_window_ready(window_rows, window_ts)

    def set_features_sink(self, sink: Callable[[dict, float], None]) -> None:
        """ Registra l'istanza del DrowsinessClassifier come consumatore dei dati """
        self._features_sink = sink

    def _on_window_ready(self, window_rows: List[Tuple[float, ...]], window_ts: List[float]) -> None:
        self._windows_emitted += 1

        if not self._calibrated and self._windows_emitted == 1:
            log_system("[DataBuffer] Warmup window #1 dropped to stabilize streams.")
            self._calibrated = True
            return

        # Estrazione canali assi. Accelerometro riscalato in unità 'g' (mg / 1000.0)
        accX = np.asarray([r[0] for r in window_rows], dtype=np.float32) / 1000.0
        accZ = np.asarray([r[2] for r in window_rows], dtype=np.float32) / 1000.0
        gyrX = np.asarray([r[3] for r in window_rows], dtype=np.float32)
        ts_array = np.asarray(window_ts, dtype=np.float64)

        window_end_ts = float(window_ts[-1]) if window_ts else 0.0

        if self._debug_print_buffer:
            log_system(f"[DataBuffer] [DEBUG] Emitting window context at ts {window_end_ts}")

        # Incapsulamento nel dizionario strutturato per la pipeline di classificazione
        window_payload = {
            'accX': accX,
            'accZ': accZ,
            'gyrX': gyrX,
            'ts_array': ts_array,
            'hop_size': self.hop_size
        }

        if self._features_sink is not None:
            try:
                self._features_sink(window_payload, window_end_ts)
            except Exception as e:
                log_system(f"[DataBuffer] Error inside DrowsinessClassifier: {type(e).__name__}: {e}", level="ERROR")

    def is_calibrated(self) -> bool:
        return self._calibrated