"""
=============================================================================
 DrowsinessClassifier — Real-Time Drowsiness Detection
 Sensor  : STM BlueCoin CPSA_L2 (LSM6DSM), single right device
 Platform: Kria KV260 / Ubuntu 22.04
=============================================================================

 DETECTED EVENTS:
   1. SLOW DRIFT   — gradual forward head tilt  (sustained Δθ > threshold)
   3. SUDDEN DROP  — abrupt forward head fall   (|ω| high, excursion rapid)

 DATA FLOW:
   DataBuffer (sliding window 150 samples, hop 75)
     └─► recognize(window_data, window_end_ts)
           └─► complementary filter  [sample-by-sample, cross-window state]
           └─► dynamic baseline      [updated every second, guard ±5°]
           └─► state machine         [SUDDEN_DROP → SLOW_DRIFT → 0]
           └─► event_queue           [consumed by EventDispatcher]

 INPUT (window_data dict):
   accX      : np.ndarray [window_size]  — pitch axis accel, in g (mg/1000)
   accZ      : np.ndarray [window_size]  — vertical axis accel, in g
   gyrX      : np.ndarray [window_size]  — pitch angular velocity, in dps
   ts_array  : np.ndarray [window_size]  — monotonic timestamps per sample
   hop_size  : int                       — new samples in this window

 OUTPUT (event dict, enqueued every window):
   id            : uuid hex string
   timestamp     : ISO wall-clock string (for logging only)
   window_ts     : float, monotonic timestamp of last sample in window
   source        : "bc_right"
   features      : [theta_last, omega_last]  — last sample of the window
   drowsiness_tag: str "0" | "1" | "3"

=============================================================================
"""

import uuid
import collections
from datetime import datetime
import numpy as np

from utils.logger import log_system, log_event
from utils.event_queue import enqueue_drop_oldest, get_event_queue


# =============================================================================
#  ██████╗  █████╗ ██████╗  █████╗ ███╗   ███╗███████╗████████╗███████╗██████╗ ███████╗
#  ██╔══██╗██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝
#  ██████╔╝███████║██████╔╝███████║██╔████╔██║█████╗     ██║   █████╗  ██████╔╝███████╗
#  ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝     ██║   ██╔══╝  ██╔══██╗╚════██║
#  ██║     ██║  ██║██║  ██║██║  ██║██║ ╚═╝ ██║███████╗   ██║   ███████╗██║  ██║███████║
#  ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝
#
#  Tune these parameters to adapt detection to your session.
#  Each parameter is annotated with its unit and tuning direction.
#
# =============================================================================

# --- COMPLEMENTARY FILTER ----------------------------------------------------
# Alpha: gyroscope weight in the fusion  (0 < alpha < 1)
#   High values (0.97–0.99) → trusts gyro more
#     PRO: fast response to quick movements
#     CON: gyro drift accumulates slowly and is never fully corrected
#   Low values  (0.90–0.95) → stronger accelerometer correction
#     PRO: less long-term drift
#     CON: more sensitive to vehicle vibrations and braking
# Recommended starting point: 0.96
CF_ALPHA = 0.96

# Accelerometer gating: if |a_total - 1g| > this value [g],
# the accelerometer correction term is disabled (vehicle accelerating/braking).
# Increase → correction active even during mild vehicle motion
# Decrease → more conservative gating, ignores accel in more situations
ACCEL_GATE_THRESHOLD_G = 0.15   # [g]

# --- BASELINE AND CALIBRATION ------------------------------------------------
# Rolling window for dynamic baseline estimation [seconds].
# Baseline too short → it follows the drowsiness itself, events are missed
# Baseline too long  → slow to compensate intentional posture changes
BASELINE_WINDOW_SEC = 30.0

# Guard band for baseline update: samples with |Δθ| > this are excluded [°].
# Only samples close to the current baseline contribute to the rolling average,
# preventing drowsiness episodes from shifting the reference point.
BASELINE_GUARD_DEG = 5.0

# --- EVENT 3: SUDDEN DROP ----------------------------------------------------
# Angular velocity threshold to qualify the drop as "fast" [dps]
# Increase → fewer false positives from road bumps
# Decrease → detects lighter sudden drops
SUDDEN_DROP_GYRO_THRESH = 10.0  # [dps]

# Minimum angular excursion from baseline to register the event [°]
# (max_delta − min_delta over the analysis window must exceed this)
SUDDEN_DROP_ANGLE_THRESH = 10.0  # [°]

# Time window within which the drop must occur [seconds]
# (cadence: sudden_samples = int(window_sec * target_fs) samples analysed)
SUDDEN_DROP_WINDOW_SEC = 0.40   # [s]

# --- EVENT 1: SLOW DRIFT -----------------------------------------------------
# Deviation from baseline beyond which the drift counter starts [°]
SLOW_DRIFT_ANGLE_THRESH = 10.0  # [°]

# Minimum time the head must remain above threshold to fire the alert [seconds]
# Increase → less sensitive to brief intentional head movements
# Decrease → earlier alert, but more false positives on long road curves
SLOW_DRIFT_DURATION_SEC = 1.5   # [s]

# Maximum angular velocity allowed during the drift [dps].
# Distinguishes drowsy drift (slow) from intentional fast movement.
SLOW_DRIFT_MAX_GYRO = 12.0      # [dps]

# --- REFRACTORY PERIOD -------------------------------------------------------
# Minimum time between any two consecutive events [seconds].
# During this window, _process_state_machine() returns the last detected tag
# rather than 0, keeping the EventDispatcher informed of the sustained state.
# Increase → fewer repeated alerts for the same episode
# Decrease → faster re-trigger if a new independent event follows quickly
REFRACTORY_SEC = 5.0            # [s]

# --- HISTORY BUFFERS ---------------------------------------------------------
# Length of the circular history deques [seconds of data at target_fs].
# Must be ≥ BASELINE_WINDOW_SEC for the baseline estimator to be meaningful.
HISTORY_WINDOW_SEC = 30.0       # [s]

# Target (nominal) sampling rate [Hz].
# Used to size sample-count windows and as dt fallback on BLE gaps.
TARGET_FS = 100.0               # [Hz]

# =============================================================================
#  END OF PARAMETERS — no need to modify anything below
# =============================================================================


class DrowsinessClassifier:
    """
    Real-time drowsiness classification via complementary filter.

    Receives sliding windows from DataBuffer, processes only the new (non-overlapping)
    samples per window, applies a cross-window complementary filter to estimate head
    pitch, and runs a two-event state machine (SUDDEN_DROP, SLOW_DRIFT).

    One structured event dict is enqueued per window regardless of the detected state,
    allowing EventDispatcher to track both alert transitions and return-to-awake.

    Thread safety: recognize() is called exclusively from the DataBuffer thread,
    so no internal locking is required.
    """

    def __init__(self, source: str = "bc_right"):
        self.source = source

        # Global queue shared with EventDispatcher — same instance, not a copy
        self.q = get_event_queue()

        # --- Copy tunable constants into instance scope ----------------------
        # Doing this allows per-instance override if needed (e.g. in tests),
        # while the module-level constants remain the default reference.
        self.alpha                 = CF_ALPHA
        self.gate_thresh_g         = ACCEL_GATE_THRESHOLD_G
        self.target_fs             = TARGET_FS
        self.sudden_drop_gyro_thresh  = SUDDEN_DROP_GYRO_THRESH
        self.sudden_drop_angle_thresh = SUDDEN_DROP_ANGLE_THRESH
        self.slow_drift_angle_thresh  = SLOW_DRIFT_ANGLE_THRESH
        self.slow_drift_max_gyro      = SLOW_DRIFT_MAX_GYRO

        # --- Complementary filter state  -------------------------------------
        self.theta_prev  = 0.0   # pitch angle estimated at the previous sample [°]
        self.baseline    = 0.0   # current neutral-posture reference angle [°]
        self._calibrated = False  # True after the first sample initialises the filter

        # --- Circular history deques -----------------------------------------
        # All three are appended in sync; maxlen enforces the rolling window.
        _max_samples = int(HISTORY_WINDOW_SEC * self.target_fs)  # e.g. 3000
        self.theta_history = collections.deque(maxlen=_max_samples)  # estimated pitch [°]
        self.gyro_history  = collections.deque(maxlen=_max_samples)  # raw ω [dps]
        self.time_history  = collections.deque(maxlen=_max_samples)  # monotonic ts [s]

        # --- Baseline update timer -------------------------------------------
        self.last_baseline_update_time = 0.0   # monotonic ts of last baseline recalc

        # --- State machine states --------------------------------------------
        self.in_drift       = False   # True while inside a potential SLOW_DRIFT episode
        self.drift_start_time = 0.0   # monotonic ts when the drift condition first held

        self.last_event_time = -999.0  # monotonic ts of the last fired event;
                                        # initialised far in the past so the first
                                        # sample is never inside the refractory window

        # --- Cross-window deduplication --------------------------------------
        self.last_processed_ts = 0.0   # window_end_ts of the last processed window;
                                        # 0.0 means "no window has been processed yet"
        self._current_tag = 0          # tag produced by the last complete window

        log_system("[DrowsinessClassifier] Real-time pipeline initialised and aligned to EventDispatcher.")


    # =========================================================================
    #  PUBLIC INTERFACE — called by DataBuffer._on_window_ready()
    # =========================================================================

    def recognize(self, window_data: dict, window_end_ts: float):
        """
        Sink called by DataBuffer every time a new sliding window is ready.

        Processes only the hop_size new samples (non-overlapping tail of the window),
        runs the complementary filter and state machine on each, then enqueues one
        event to the global queue.

        Args:
            window_data   : dict with keys accX, accZ, gyrX, ts_array, hop_size
            window_end_ts : monotonic timestamp of the last sample in the window

        Returns:
            The event dict just enqueued, or None on error.
        """

        # --- 0. Unpack and validate window payload ---------------------------
        try:
            accX     = window_data['accX']      # pitch-axis accel [g], shape (window_size,)
            accZ     = window_data['accZ']      # vertical-axis accel [g]
            gyrX     = window_data['gyrX']      # pitch angular velocity [dps]
            ts_array = window_data['ts_array']  # per-sample monotonic timestamps [s]
            hop_size = window_data['hop_size']  # number of genuinely new samples
        except KeyError as e:
            log_system(f"[Classifier] Missing expected key in window_data: {e}", level="ERROR")
            return None

        # --- 1. Determine which samples are genuinely new --------------------
        # The sliding window overlaps with the previous one by (window_size - hop_size)
        # samples. We skip those already-processed samples to avoid double-counting
        # them in the filter integration and baseline update.
        if self.last_processed_ts == 0.0:
            start_idx = 0           # first window ever: process all samples
        else:
            start_idx = len(accX) - hop_size   # subsequent windows: only the new tail

        if start_idx < 0 or start_idx >= len(accX):
            return None   # guard against misconfigured hop_size

        # --- 2. Per-sample processing loop -----------------------------------
        # Accumulates the most severe tag seen across new samples in this window.
        detected_tag_this_window = 0   # default: AWAKE

        try:
            for i in range(start_idx, len(accX)):
                ap_g  = accX[i]       # pitch-axis accel component [g]
                av_g  = accZ[i]       # vertical-axis accel component [g]
                omega = gyrX[i]       # angular velocity around pitch axis [dps]
                t_now = ts_array[i]   # monotonic timestamp of this sample [s]

                # ── dt calculation ───────────────────────────────────────────
                # Use the real inter-sample interval from BLE timestamps.
                # Fallback to 1/target_fs on first sample or on anomalous gaps
                # (gap ≤ 0 signals duplicate/out-of-order packets; gap > 100 ms
                # signals a BLE dropout that would cause a large gyro integration
                # spike if used directly).
                if len(self.time_history) > 0:
                    dt = t_now - self.time_history[-1]
                    if dt <= 0 or dt > 0.1:
                        dt = 1.0 / self.target_fs   # [s] — nominal period fallback
                else:
                    dt = 1.0 / self.target_fs

                # ── Complementary filter — step 1: accelerometer estimate ────
                # atan2(ap, |av|) gives the angle between the horizontal and the
                # gravity vector projected on the pitch plane.
                # Accurate long-term, noisy short-term.
                theta_accel = np.degrees(np.arctan2(ap_g, abs(av_g)))   # [°]

                # ── Lazy calibration on first sample ─────────────────────────
                # Initialises theta_prev and baseline from the first real reading
                # rather than from the placeholder 0.0 set in __init__.
                if not self._calibrated:
                    self.theta_prev              = theta_accel
                    self.baseline                = theta_accel
                    self._calibrated             = True
                    self.last_baseline_update_time = t_now

                # ── Complementary filter — step 2: gyroscope prediction ──────
                # Integrate angular velocity forward: θ_gyro = θ_prev + ω·dt
                # Accurate short-term, drifts over minutes.
                theta_gyro_pred = self.theta_prev + (omega * dt)   # [°]

                # ── Complementary filter — step 3: gating + fusion ──────────
                # Disable the accelerometer correction term when the vehicle is
                # accelerating or braking (|a_total - 1g| > gate_thresh_g):
                # in those conditions the measured acceleration is not pure gravity,
                # so atan2 would return a wrong angle.
                a_total_g = np.sqrt(ap_g**2 + av_g**2)         # [g], should be ≈1 when static
                is_static = abs(a_total_g - 1.0) < self.gate_thresh_g

                if is_static:
                    # Fusion: weight gyro prediction (alpha) + accel estimate (1-alpha)
                    theta = self.alpha * theta_gyro_pred + (1.0 - self.alpha) * theta_accel
                else:
                    # Vehicle in motion: trust gyro only, ignore accel this sample
                    theta = theta_gyro_pred

                self.theta_prev = theta   # propagate to next iteration

                # ── Append to circular histories ─────────────────────────────
                # All three deques share the same logical index.
                # maxlen ensures old data falls off automatically.
                self.theta_history.append(theta)
                self.gyro_history.append(omega)
                self.time_history.append(t_now)

                # ── Dynamic baseline update (at most once per second) ────────
                # Recomputing the mean over 3000 items at 100 Hz would be wasteful;
                # once per second is enough given how slowly posture drifts.
                if t_now - self.last_baseline_update_time >= 1.0:
                    self.last_baseline_update_time = t_now
                    theta_list = list(self.theta_history)
                    if len(theta_list) > 10:
                        # Only samples within ±BASELINE_GUARD_DEG of the current
                        # baseline vote for the new one. This excludes drowsiness
                        # episodes from pulling the reference point along with them.
                        close_mask = [abs(th - self.baseline) < BASELINE_GUARD_DEG
                                      for th in theta_list]
                        if sum(close_mask) > len(theta_list) // 4:
                            self.baseline = float(np.mean(
                                [theta_list[idx] for idx in range(len(theta_list))
                                 if close_mask[idx]]
                            ))

                # ── State machine ────────────────────────────────────────────
                # Returns the tag for this individual sample.
                # detected_tag_this_window keeps the last non-zero tag across
                # all new samples so that the window-level tag reflects the most
                # recently detected event rather than just the last sample state.
                tag_sample = self._process_state_machine(theta, omega, t_now)
                if tag_sample != 0:
                    detected_tag_this_window = tag_sample

        except Exception as e:
            log_system(f"[Classifier] Processing execution error: {type(e).__name__}: {e}",
                       level="ERROR")
            return None

        # --- 3. Finalise window state ----------------------------------------
        self.last_processed_ts = window_end_ts   # mark this window as consumed
        self._current_tag = detected_tag_this_window

        # --- 4. Build structured event dict ----------------------------------
        # theta and omega here are the values from the last sample of the loop.
        # They are stored in "features" for lightweight diagnostic logging only;
        # the authoritative classification output is drowsiness_tag.
        try:
            event = {
                "id"            : uuid.uuid4().hex,
                "timestamp"     : datetime.now().isoformat(),  # wall-clock, for logs
                "window_ts"     : float(window_end_ts) if window_end_ts is not None else None,
                "source"        : self.source,
                "features"      : [float(theta), float(omega)],  # last sample of window
                "drowsiness_tag": str(self._current_tag),         # "0", "1", or "3"
            }
        except Exception as e:
            log_system(f"[Classifier] Serialization error: {type(e).__name__}: {e}", level="ERROR")
            return None

        # --- 5. Enqueue to global event queue --------------------------------
        # enqueue_drop_oldest() inserts the event and, if the queue is full,
        # pops and returns the oldest item to make room (FIFO with drop).
        # With event_queue_size=1 (config.yaml), the queue fills after every
        # window the dispatcher hasn't consumed yet — expect frequent drops
        # under high IMU rate unless the dispatcher is fast enough.
        dropped, dropped_item = enqueue_drop_oldest(self.q, event, kind="imu")
        if dropped:
            log_system("[Classifier] Oldest event dropped (queue full)", level="WARNING")
            try:
                ev = dropped_item or {}
                log_event(
                    timestamp    = ev.get("timestamp", event["timestamp"]),
                    feature_type = ev.get("type", "imu"),
                    event        = ev.get("drowsiness_tag", "UNKNOWN"),
                    actuations   = [{"target": "DROPPED", "params": {"reason": "queue_full"}}],
                    source       = ev.get("source", self.source),
                )
            except Exception:
                pass   # logging the dropped item must never block the pipeline

        return event


    # =========================================================================
    #  PRIVATE — state machine
    # =========================================================================

    def _process_state_machine(self, theta: float, omega: float, t_now: float) -> int:
        """
        Two-pattern drowsiness state machine evaluated on every new sample.

        Tag mapping (aligned with EventDispatcher LABELS):
          0 → AWAKE        (default, no anomaly)
          1 → SLOW_DRIFT   (head drifting forward, sustained)
          3 → SUDDEN_DROP  (abrupt forward head fall)

        Returns the detected tag for this sample, or _current_tag during the
        refractory window following a fired event.
        """

        # ── Refractory gate ──────────────────────────────────────────────────
        # After any event fires, hold _current_tag for REFRACTORY_SEC seconds.
        # This prevents multiple re-triggers on the same physical episode and
        # keeps the downstream dispatcher informed of the sustained alert state.
        if t_now - self.last_event_time < REFRACTORY_SEC:
            return self._current_tag

        delta     = theta - self.baseline   # signed deviation from neutral posture [°]
        abs_omega = abs(omega)              # unsigned angular velocity [dps]

        # ── EVENT 3: SUDDEN DROP ─────────────────────────────────────────────
        # Evaluated first — highest priority; a confirmed drop resets all other states.
        #
        # Detection logic: look back over the last SUDDEN_DROP_WINDOW_SEC seconds
        # and require four simultaneous conditions to be met:
        #   (a) current deviation > angle threshold    — head is actually forward
        #   (b) current |ω| > gyro threshold           — currently moving fast
        #   (c) peak |ω| in window > gyro threshold    — peak happened in the window
        #   (d) excursion (max_Δθ − min_Δθ) > angle threshold — rapid excursion occurred
        # Condition (d) is the most discriminating: it rules out a sustained
        # static tilt (which satisfies a and b but not d).
        sudden_samples = int(SUDDEN_DROP_WINDOW_SEC * self.target_fs)  # e.g. 40 samples
        if len(self.theta_history) >= sudden_samples:
            recent_theta = list(self.theta_history)[-sudden_samples:]
            recent_delta = [th - self.baseline for th in recent_theta]
            recent_gyro  = [abs(g) for g in list(self.gyro_history)[-sudden_samples:]]

            if (delta > self.sudden_drop_angle_thresh               # (a)
                    and abs_omega > self.sudden_drop_gyro_thresh     # (b)
                    and max(recent_gyro) > self.sudden_drop_gyro_thresh  # (c)
                    and (max(recent_delta) - min(recent_delta)) > self.sudden_drop_angle_thresh):  # (d)

                log_system(
                    f"🚨 [SUDDEN DROP] Head dropped abruptly! "
                    f"t={t_now:.2f}s  Δθ={delta:.1f}°  ω={omega:.1f}dps",
                    level="WARNING"
                )
                self.last_event_time = t_now   # start refractory window
                self.in_drift        = False   # cancel any ongoing drift episode
                return 3

        # ── EVENT 1: SLOW DRIFT ──────────────────────────────────────────────
        # Two-phase: entry condition must hold continuously for SLOW_DRIFT_DURATION_SEC.
        #
        # Entry condition:
        #   (a) deviation > angle threshold            — head is forward
        #   (b) |ω| < slow_drift_max_gyro              — movement is slow (not a drop)
        #
        # If the condition breaks at any point before the timer expires, the
        # episode is discarded and in_drift resets — the timer does not pause.
        if delta > self.slow_drift_angle_thresh and abs_omega < self.slow_drift_max_gyro:
            if not self.in_drift:
                # Entry: start timing the episode
                self.in_drift        = True
                self.drift_start_time = t_now
            elif (t_now - self.drift_start_time) >= SLOW_DRIFT_DURATION_SEC:
                # Threshold sustained long enough → fire the event
                log_system(
                    f"🚨 [SLOW DRIFT] Head tilted forward continuously! "
                    f"t={t_now:.2f}s  Δθ={delta:.1f}°",
                    level="WARNING"
                )
                self.last_event_time = t_now   # start refractory window
                self.in_drift        = False   # reset for next potential episode
                return 1
        else:
            # Condition broke — discard episode, reset timer
            self.in_drift = False

        return 0   # no event detected this sample → AWAKE
