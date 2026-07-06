# IMU_pipeline/classifiers/drowsiness_classifier.py
# Online streaming classifier for drowsiness detection via complementary filter.
# Aligned to process sample-by-sample subwindows streaming from the DataBuffer.
#
# Adapted for Drowsiness Detection - CPSA_2026

import uuid
import collections
from datetime import datetime
import numpy as np

from utils.logger import log_system, log_event
from utils.event_queue import enqueue_drop_oldest, get_event_queue

class DrowsinessClassifier:
    """
    Real-time adaptation of the drowsiness detection pipeline.
    Maintains cross-window filtering state, executes pattern logic,
    and enqueues recognition events for actuation (EventDispatcher compatible).
    """

    def __init__(self, source: str = "bc_right"):
        self.source = source
        self.q = get_event_queue()  # Global queue shared with EventDispatcher
        
        # Complementary filter tuning parameters
        self.alpha = 0.96
        self.gate_thresh_g = 0.15
        self.target_fs = 100.0  # Estimated/Target ODR
        
        # State machine event thresholds
        self.sudden_drop_gyro_thresh = 10.0
        self.sudden_drop_angle_thresh = 10.0
        self.slow_drift_angle_thresh = 10.0
        self.slow_drift_max_gyro = 12.0
        
        # Complementary filter states and calibration
        self.theta_prev = 0.0
        self.baseline = 0.0
        self._calibrated = False
        
        # Circular history buffers (up to 30 seconds of data at 100Hz for baseline estimation)
        max_history_samples = int(30.0 * self.target_fs)
        self.theta_history = collections.deque(maxlen=max_history_samples)
        self.gyro_history = collections.deque(maxlen=max_history_samples)
        self.time_history = collections.deque(maxlen=max_history_samples)
        
        # Internal states for the event state machine
        self.in_drift = False
        self.drift_start_time = 0.0
        
        self.last_baseline_update_time = 0.0
        self.last_event_time = -999.0

        # Internal flag to distinguish new samples from sliding window overlaps
        self.last_processed_ts = 0.0
        self._current_tag = 0

        log_system("[DrowsinessClassifier] Real-time pipeline initialized and aligned to EventDispatcher.")

    def recognize(self, window_data: dict, window_end_ts: float):
        """
        Sink interface called by DataBuffer upon every window emission.
        Args:
            window_data (dict): Contains vectors accX, accZ, gyrX, ts_array, hop_size
            window_end_ts (float): Closing timestamp of the current window
        """
        try:
            accX = window_data['accX']
            accZ = window_data['accZ']
            gyrX = window_data['gyrX']
            ts_array = window_data['ts_array']
            hop_size = window_data['hop_size']
        except KeyError as e:
            log_system(f"[Classifier] Missing expected key in window_data: {e}", level="ERROR")
            return None

        # Prevent reprocessing overlapping samples resulting from the sliding window hop
        if self.last_processed_ts == 0.0:
            start_idx = 0
        else:
            start_idx = len(accX) - hop_size

        if start_idx < 0 or start_idx >= len(accX):
            return None

        # Default to 0 (Awake); updates if the state machine intercepts anomalies
        detected_tag_this_window = 0

        try:
            for i in range(start_idx, len(accX)):
                ap_g = accX[i]
                av_g = accZ[i]
                omega = gyrX[i]
                t_now = ts_array[i]

                # Dynamic dt calculation based on real timestamps
                if len(self.time_history) > 0:
                    dt = t_now - self.time_history[-1]
                    if dt <= 0 or dt > 0.1:  # Safety fallback for temporal gaps
                        dt = 1.0 / self.target_fs
                else:
                    dt = 1.0 / self.target_fs

                # 1. Complementary Filter
                theta_accel = np.degrees(np.arctan2(ap_g, abs(av_g)))
                
                if not self._calibrated:
                    self.theta_prev = theta_accel
                    self.baseline = theta_accel
                    self._calibrated = True
                    self.last_baseline_update_time = t_now

                theta_gyro_pred = self.theta_prev + (omega * dt)
                a_total_g = np.sqrt(ap_g**2 + av_g**2)
                is_static = abs(a_total_g - 1.0) < self.gate_thresh_g

                if is_static:
                    theta = self.alpha * theta_gyro_pred + (1.0 - self.alpha) * theta_accel
                else:
                    theta = theta_gyro_pred

                self.theta_prev = theta

                # Store in current history
                self.theta_history.append(theta)
                self.gyro_history.append(omega)
                self.time_history.append(t_now)

                # 2. Dynamic Baseline Update (Every second)
                if t_now - self.last_baseline_update_time >= 1.0:
                    self.last_baseline_update_time = t_now
                    theta_list = list(self.theta_history)
                    if len(theta_list) > 10:
                        close_mask = [abs(th - self.baseline) < 5.0 for th in theta_list]
                        if sum(close_mask) > len(theta_list) // 4:
                            self.baseline = float(np.mean([theta_list[idx] for idx in range(len(theta_list)) if close_mask[idx]]))

                # 3. State Machine Analysis for aligned drowsiness tag extraction
                tag_sample = self._process_state_machine(theta, omega, t_now)
                if tag_sample != 0:
                    detected_tag_this_window = tag_sample

        except Exception as e:
            log_system(f"[Classifier] Processing execution error: {type(e).__name__}: {e}", level="ERROR")
            return None

        self.last_processed_ts = window_end_ts
        self._current_tag = detected_tag_this_window

        # Structured event generation required by the EventDispatcher
        try:
            event = {
                "id": uuid.uuid4().hex,
                "timestamp": datetime.now().isoformat(),
                "window_ts": float(window_end_ts) if window_end_ts is not None else None,
                "source": self.source,
                "features": [float(theta), float(omega)], # Keep minimal track of angle and gyroscope
                "drowsiness_tag": str(self._current_tag)   # String tag (0, 1, 2, 3) consumed by the video core
            }
        except Exception as e:
            log_system(f"[Classifier] Serialization error: {type(e).__name__}: {e}", level="ERROR")
            return None

        # Asynchronous insertion into the global thread-safe queue (handles full queues by dropping the oldest)
        dropped, dropped_item = enqueue_drop_oldest(self.q, event, kind="imu")
        if dropped:
            log_system("[Classifier] Oldest event dropped (queue full)", level="WARNING")
            try:
                ev = dropped_item or {}
                log_event(
                    timestamp=ev.get("timestamp", event["timestamp"]),
                    feature_type=ev.get("type", "imu"),
                    event=ev.get("drowsiness_tag", "UNKNOWN"),
                    actuations=[{"target": "DROPPED", "params": {"reason": "queue_full"}}],
                    source=ev.get("source", self.source),
                )
            except Exception:
                pass

        return event

    def _process_state_machine(self, theta: float, omega: float, t_now: float) -> int:
        """
        Drowsiness detection with native mapping for the EventDispatcher:
          0 -> Awake, 1 -> SLOW_DRIFT, 2 -> NOD, 3 -> SUDDEN_DROP
        """
        if t_now - self.last_event_time < 5.0:
            return self._current_tag

        delta = theta - self.baseline
        abs_omega = abs(omega)

        # ── EVENT 2: SUDDEN HEAD DROP (Maximum Danger) ───────────
        sudden_samples = int(0.40 * self.target_fs)
        if len(self.theta_history) >= sudden_samples:
            recent_theta = list(self.theta_history)[-sudden_samples:]
            recent_delta = [th - self.baseline for th in recent_theta]
            recent_gyro = [abs(g) for g in list(self.gyro_history)[-sudden_samples:]]

            if (delta > self.sudden_drop_angle_thresh 
                    and abs_omega > self.sudden_drop_gyro_thresh 
                    and max(recent_gyro) > self.sudden_drop_gyro_thresh 
                    and (max(recent_delta) - min(recent_delta)) > self.sudden_drop_angle_thresh):
                
                log_system(f"🚨 [SUDDEN DROP] Head dropped abruptly! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                self.last_event_time = t_now
                self.in_drift = False
                return 2  # Aligned with LABELS[2] = "SUDDEN_DROP"

        # ── EVENT 1: SLOW HEAD DRIFT ─────────────────────────────
        if delta > self.slow_drift_angle_thresh and abs_omega < self.slow_drift_max_gyro:
            if not self.in_drift:
                self.in_drift = True
                self.drift_start_time = t_now
            elif (t_now - self.drift_start_time) >= 1.5:
                log_system(f"🚨 [SLOW DRIFT] Head tilted forward continuously! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                self.last_event_time = t_now
                self.in_drift = False
                return 1  # Aligned with LABELS[1] = "SLOW_DRIFT"
        else:
            self.in_drift = False

        return 0
