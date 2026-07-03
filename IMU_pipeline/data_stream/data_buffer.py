"""
=============================================================================
 DataBuffer — Sliding Window Buffer for Real-Time IMU Streaming
 Source  : IMUSynchronizer (single device, bc_right)
 Sink    : DrowsinessClassifier.recognize()
=============================================================================

 ROLE IN THE PIPELINE:
   IMUSynchronizer
     └─► add_buffer_row(R_acc, R_gyr, ts_emit)   [called on every acc+gyr pair]
           └─► accumulates rows until window_size is reached
           └─► emits window → _on_window_ready()
                 └─► extracts axes, scales units, builds payload dict
                 └─► _features_sink(window_payload, window_end_ts)
                       └─► DrowsinessClassifier.recognize()

 SLIDING WINDOW MECHANICS:
   window_size = 150 samples   (config.yaml: buffer.window_size)
   hop_size    =  75 samples   (config.yaml: buffer.overlap)
   overlap     =  75 samples   (window_size − hop_size = 50%)

   After each emission, _rows is trimmed by hop_size from the front,
   so the next window shares the last 75 samples with the current one.
   The classifier handles overlap deduplication internally via start_idx.

 WARMUP:
   The very first window is always dropped silently.
   BLE streams are not fully stable immediately after connection;
   discarding window #1 prevents the filter from calibrating on
   potentially noisy or incomplete data.

 OUTPUT PAYLOAD (window_payload dict):
   accX     : np.ndarray float32 [window_size]  — pitch-axis accel   [g]
   accZ     : np.ndarray float32 [window_size]  — vertical-axis accel [g]
   gyrX     : np.ndarray float32 [window_size]  — pitch angular vel.  [dps]
   ts_array : np.ndarray float64 [window_size]  — monotonic timestamps [s]
   hop_size : int                               — new samples in window

=============================================================================
"""

from __future__ import annotations

from typing import List, Tuple, Callable, Optional
import threading
import numpy as np

from utils.logger import log_system
from utils.config import get_buffer_config

# Type alias for a 3-axis sensor reading (x, y, z)
Vec3 = Tuple[float, float, float]


class DataBuffer:
    """
    Accumulates aligned (acc, gyr) rows from the IMUSynchronizer and emits
    sliding windows to the DrowsinessClassifier.

    Each internal row stores all 6 raw axis values as floats.
    Unit conversion (mg → g) is deferred to _on_window_ready() so that
    the storage layer stays unit-agnostic and close to the wire format.

    Thread safety: add_buffer_row() is protected by a Lock because it is
    called from the BLE callback thread, while set_features_sink() may be
    called from the main thread during initialisation.
    """

    def __init__(self, window_size: Optional[int] = None, hop_size: Optional[int] = None):
        """
        Args:
            window_size : samples per emitted window; overrides config if provided.
            hop_size    : samples to advance after each emission (= new samples
                          per window, i.e. window_size − overlap); overrides config.
        """
        cfg = get_buffer_config() or {}

        # Number of samples that must accumulate before a window is emitted.
        # At 100 Hz, 150 samples ≈ 1.5 seconds of data per window.
        self.window_size = int(window_size if window_size is not None
                               else cfg.get("window_size", 150))

        # Advance step after each emission.
        # hop_size = 75 → 50% overlap → window slides by 0.75 s at 100 Hz.
        self.hop_size = int(hop_size if hop_size is not None
                            else cfg.get("overlap", 75))

        # Enable verbose per-window debug logging (high console load; off by default)
        self._debug_print_buffer = bool(cfg.get("debug_print_buffer", False))

        # --- Internal storage ------------------------------------------------
        # _rows and _ts grow until window_size is reached, then are trimmed
        # by hop_size from the front after each emission. Both lists are always
        # the same length and share the same logical index.
        self._rows: List[Tuple[float, ...]] = []   # raw 6-float rows (acc_xyz, gyr_xyz)
        self._ts:   List[float] = []               # monotonic timestamp per row [s]
        self._lock  = threading.Lock()             # guards _rows and _ts

        # --- Classifier sink -------------------------------------------------
        # Set via set_features_sink(); called once per emitted window.
        # None until explicitly registered — windows are silently dropped if unset.
        self._features_sink: Optional[Callable[[dict, float], None]] = None

        # --- Warmup state ----------------------------------------------------
        self._windows_emitted = 0    # total windows generated since startup
        self._calibrated      = False  # True after window #1 is dropped (warmup done)

        log_system(f"[DataBuffer] Pipeline Ready: window={self.window_size}  hop={self.hop_size}")


    # =========================================================================
    #  PUBLIC INTERFACE
    # =========================================================================

    def add_buffer_row(self, R_acc: Vec3, R_gyr: Vec3, ts_emit: float) -> None:
        """
        Append one aligned (acc, gyr) sample to internal storage.

        Called by IMUSynchronizer every time a new acc+gyr pair is ready.
        If appending this row causes the buffer to reach window_size, a window
        is extracted and emitted; the buffer then slides forward by hop_size.

        The classifier call (_on_window_ready) is intentionally placed OUTSIDE
        the lock scope: holding the lock during classification would block
        incoming BLE callbacks for the duration of the classifier's execution.

        Args:
            R_acc    : (ax, ay, az) from the right BlueCoin, in mg
            R_gyr    : (gx, gy, gz) from the right BlueCoin, in dps
            ts_emit  : monotonic timestamp of this sample [s]
        """
        window_rows = None
        window_ts   = None

        # Flatten the two Vec3 tuples into a single 6-float row.
        # Index layout:  0=acc_x  1=acc_y  2=acc_z  3=gyr_x  4=gyr_y  5=gyr_z
        row = (
            float(R_acc[0]), float(R_acc[1]), float(R_acc[2]),   # acc_x, acc_y, acc_z  [mg]
            float(R_gyr[0]), float(R_gyr[1]), float(R_gyr[2]),   # gyr_x, gyr_y, gyr_z  [dps]
        )

        with self._lock:
            self._rows.append(row)
            self._ts.append(float(ts_emit))

            if len(self._rows) >= self.window_size:
                # ── Extract the window ────────────────────────────────────────
                # Take the last window_size rows. With the current sliding logic,
                # start is always 0 the first time (len == window_size) and then
                # equals len(_rows) - window_size on subsequent calls, which also
                # resolves to 0 after the trim — kept explicit for clarity.
                start       = len(self._rows) - self.window_size
                window_rows = self._rows[start : start + self.window_size]
                window_ts   = self._ts[start   : start + self.window_size]

                # ── Slide forward by hop_size ─────────────────────────────────
                # Discard the oldest hop_size rows so the buffer retains the
                # last (window_size − hop_size) rows as overlap for the next window.
                if self.hop_size <= 0:
                    # Edge case: hop_size = 0 would mean infinite overlap; clear all.
                    self._rows.clear()
                    self._ts.clear()
                else:
                    self._rows = self._rows[self.hop_size:]
                    self._ts   = self._ts[self.hop_size:]

        # ── Emit outside the lock ─────────────────────────────────────────────
        # Classification can take several milliseconds; releasing the lock first
        # ensures BLE callbacks are never queued behind classifier execution.
        if window_rows is not None:
            self._on_window_ready(window_rows, window_ts)

    def set_features_sink(self, sink: Callable[[dict, float], None]) -> None:
        """
        Register the callable that will receive each emitted window.

        Expected signature:  sink(window_payload: dict, window_end_ts: float)
        In the current pipeline this is always DrowsinessClassifier.recognize().
        Must be called before the first window is emitted, or early windows
        will be silently dropped (the _features_sink is None guard in
        _on_window_ready handles this safely).
        """
        self._features_sink = sink

    def is_calibrated(self) -> bool:
        """
        Returns True once the warmup window has been dropped and the buffer
        is ready to send meaningful data to the classifier.
        """
        return self._calibrated


    # =========================================================================
    #  PRIVATE
    # =========================================================================

    def _on_window_ready(self,
                         window_rows: List[Tuple[float, ...]],
                         window_ts:   List[float]) -> None:
        """
        Called internally every time a full window has been extracted.
        Handles warmup, unit conversion, payload construction, and sink dispatch.

        Args:
            window_rows : list of window_size 6-float rows (raw, in mg / dps)
            window_ts   : list of window_size monotonic timestamps [s]
        """
        self._windows_emitted += 1

        # ── Warmup: drop window #1 ────────────────────────────────────────────
        # The BLE stream is not fully stable immediately after connection.
        # Discarding the first window prevents the complementary filter in
        # DrowsinessClassifier from calibrating its baseline on potentially
        # incomplete or noisy samples.
        if not self._calibrated and self._windows_emitted == 1:
            log_system("[DataBuffer] Warmup window #1 dropped to stabilize streams.")
            self._calibrated = True
            return

        # ── Axis extraction and unit conversion ───────────────────────────────
        # Only the three axes consumed by the classifier are extracted.
        # acc_y, gyr_y, gyr_z are stored in _rows but not forwarded;
        # the remaining axes are available here if needed in the future.
        #
        # Row index layout:  0=acc_x  1=acc_y  2=acc_z  3=gyr_x  4=gyr_y  5=gyr_z
        #
        # Accelerometer: mg → g (divide by 1000).
        # The classifier expects g so that atan2 and the gating threshold
        # (|a_total - 1g|) work on the correct numeric scale.
        accX = np.asarray([r[0] for r in window_rows], dtype=np.float32) / 1000.0  # pitch axis  [g]
        accZ = np.asarray([r[2] for r in window_rows], dtype=np.float32) / 1000.0  # vertical    [g]
        gyrX = np.asarray([r[3] for r in window_rows], dtype=np.float32)            # pitch ω     [dps]

        # Timestamps kept as float64 to preserve the full monotonic precision.
        ts_array = np.asarray(window_ts, dtype=np.float64)   # [s]

        # Scalar end-timestamp used as the window's identity key downstream.
        window_end_ts = float(window_ts[-1]) if window_ts else 0.0

        if self._debug_print_buffer:
            log_system(f"[DataBuffer] [DEBUG] Emitting window #{self._windows_emitted} "
                       f"at ts={window_end_ts:.4f}s")

        # ── Build structured payload ──────────────────────────────────────────
        # A dict is used instead of positional arguments so that the classifier
        # interface remains explicit and order-independent; adding new channels
        # in the future only requires adding a key here and reading it there.
        window_payload = {
            'accX'    : accX,       # np.ndarray float32 [window_size]  — pitch accel [g]
            'accZ'    : accZ,       # np.ndarray float32 [window_size]  — vertical accel [g]
            'gyrX'    : gyrX,       # np.ndarray float32 [window_size]  — pitch ω [dps]
            'ts_array': ts_array,   # np.ndarray float64 [window_size]  — timestamps [s]
            'hop_size': self.hop_size,  # int — new samples in this window (for overlap skip)
        }

        # ── Dispatch to classifier ────────────────────────────────────────────
        # The sink guard (is not None) lets the buffer operate safely even if
        # set_features_sink() has not been called yet — windows are dropped
        # without error rather than raising an AttributeError.
        if self._features_sink is not None:
            try:
                self._features_sink(window_payload, window_end_ts)
            except Exception as e:
                # Catch-all: a crash inside the classifier must never propagate
                # back into the BLE callback thread and kill the acquisition loop.
                log_system(
                    f"[DataBuffer] Error inside DrowsinessClassifier: "
                    f"{type(e).__name__}: {e}",
                    level="ERROR"
                )
