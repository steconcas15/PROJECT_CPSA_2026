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
    Realtime adaptation of the drowsiness detection pipeline.
    Maintains cross-window filtering state, executes pattern logic,
    and enqueues recognition events for actuation (EventDispatcher compatible).
    """

    def __init__(self, source: str = "bc_right"):
        self.source = source
        self.q = get_event_queue()  # Coda globale condivisa con l'EventDispatcher
        
        # Parametri di Tuning del filtro complementare
        self.alpha = 0.96
        self.gate_thresh_g = 0.15
        self.target_fs = 100.0  # ODR stimata/target
        
        # Soglie eventi della macchina a stati
        self.sudden_drop_gyro_thresh = 10.0
        self.sudden_drop_angle_thresh = 10.0
        self.slow_drift_angle_thresh = 10.0
        self.slow_drift_max_gyro = 12.0
        
        # Stati del filtro complementare e calibrazione
        self.theta_prev = 0.0
        self.baseline = 0.0
        self._calibrated = False
        
        # Buffer storici circolari (fino a 30 secondi di dati a 100Hz per stima baseline)
        max_history_samples = int(30.0 * self.target_fs)
        self.theta_history = collections.deque(maxlen=max_history_samples)
        self.gyro_history = collections.deque(maxlen=max_history_samples)
        self.time_history = collections.deque(maxlen=max_history_samples)
        
        # Stati interni della macchina a stati degli eventi        
        self.in_drift = False
        self.drift_start_time = 0.0
        
        self.last_baseline_update_time = 0.0
        self.last_event_time = -999.0

        # Flag interno per discriminare i nuovi campioni dagli overlap del buffer sliding window
        self.last_processed_ts = 0.0
        self._current_tag = 0

        log_system("[DrowsinessClassifier] Realtime pipeline initialized and aligned to EventDispatcher.")

    def recognize(self, window_data: dict, window_end_ts: float):
        """
        Interfaccia di sink chiamata dal DataBuffer ad ogni emissione di finestra.
        Args:
            window_data (dict): Contiene i vettori accX, accZ, gyrX, ts_array, hop_size
            window_end_ts (float): Timestamp di chiusura della finestra corrente
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

        # Evitiamo di rielaborare i campioni sovrapposti derivanti dal salto (hop_size) della finestra sliding
        if self.last_processed_ts == 0.0:
            start_idx = 0
        else:
            start_idx = len(accX) - hop_size

        if start_idx < 0 or start_idx >= len(accX):
            return None

        # Default a 0 (Sveglio), se la macchina a stati intercetta anomalie si aggiorna
        detected_tag_this_window = 0

        try:
            for i in range(start_idx, len(accX)):
                ap_g = accX[i]
                av_g = accZ[i]
                omega = gyrX[i]
                t_now = ts_array[i]

                # Calcolo dinamico del dt basato sui timestamp reali
                if len(self.time_history) > 0:
                    dt = t_now - self.time_history[-1]
                    if dt <= 0 or dt > 0.1:  # Fallback di sicurezza per buchi temporali
                        dt = 1.0 / self.target_fs
                else:
                    dt = 1.0 / self.target_fs

                # 1. Filtro Complementare
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

                # Memorizzazione nello storico corrente
                self.theta_history.append(theta)
                self.gyro_history.append(omega)
                self.time_history.append(t_now)

                # 2. Aggiornamento baseline dinamica (Ogni secondo)
                if t_now - self.last_baseline_update_time >= 1.0:
                    self.last_baseline_update_time = t_now
                    theta_list = list(self.theta_history)
                    if len(theta_list) > 10:
                        close_mask = [abs(th - self.baseline) < 5.0 for th in theta_list]
                        if sum(close_mask) > len(theta_list) // 4:
                            self.baseline = float(np.mean([theta_list[idx] for idx in range(len(theta_list)) if close_mask[idx]]))

                # 3. Analisi della Macchina a Stati per l'estrazione dei Tag di sonnolenza aligned
                tag_sample = self._process_state_machine(theta, omega, t_now)
                if tag_sample != 0:
                    detected_tag_this_window = tag_sample

        except Exception as e:
            log_system(f"[Classifier] Processing execution error: {type(e).__name__}: {e}", level="ERROR")
            return None

        self.last_processed_ts = window_end_ts
        self._current_tag = detected_tag_this_window

        # Generazione dell'evento strutturato richiesto dall'EventDispatcher
        try:
            event = {
                "id": uuid.uuid4().hex,
                "timestamp": datetime.now().isoformat(),
                "window_ts": float(window_end_ts) if window_end_ts is not None else None,
                "source": self.source,
                "features": [float(theta), float(omega)], # Manteniamo traccia minima di angolo e giroscopio
                "stereotipy_tag": str(self._current_tag)   # Tag stringa (0, 1, 2, 3) consumato dal core video
            }
        except Exception as e:
            log_system(f"[Classifier] Serialization error: {type(e).__name__}: {e}", level="ERROR")
            return None

        # Inserimento asincrono nella coda globale thread-safe (gestisce code piene scartando il più vecchio)
        dropped, dropped_item = enqueue_drop_oldest(self.q, event, kind="imu")
        if dropped:
            log_system("[Classifier] Oldest event dropped (queue full)", level="WARNING")
            try:
                ev = dropped_item or {}
                log_event(
                    timestamp=ev.get("timestamp", event["timestamp"]),
                    feature_type=ev.get("type", "imu"),
                    event=ev.get("stereotipy_tag", "UNKNOWN"),
                    actuations=[{"target": "DROPPED", "params": {"reason": "queue_full"}}],
                    source=ev.get("source", self.source),
                )
            except Exception:
                pass

        return event

    def _process_state_machine(self, theta: float, omega: float, t_now: float) -> int:
        """
        Rilevamento sonnolenza con mappatura nativa per l'EventDispatcher:
          0 -> Sveglio, 1 -> SLOW_DRIFT, 2 -> NOD, 3 -> SUDDEN_DROP
        """
        if t_now - self.last_event_time < 5.0:
            return self._current_tag

        delta = theta - self.baseline
        abs_omega = abs(omega)

        # ── EVENTO 3: CADUTA BRUSCA della testa (Massimo pericolo) ───────────
        sudden_samples = int(0.40 * self.target_fs)
        if len(self.theta_history) >= sudden_samples:
            recent_theta = list(self.theta_history)[-sudden_samples:]
            recent_delta = [th - self.baseline for th in recent_theta]
            recent_gyro = [abs(g) for g in list(self.gyro_history)[-sudden_samples:]]

            if (delta > self.sudden_drop_angle_thresh 
                    and abs_omega > self.sudden_drop_gyro_thresh 
                    and max(recent_gyro) > self.sudden_drop_gyro_thresh 
                    and (max(recent_delta) - min(recent_delta)) > self.sudden_drop_angle_thresh):
                
                log_system(f"🚨 [SUDDEN DROP] Caduta brusca della testa! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                self.last_event_time = t_now
                self.in_drift = False
                return 3  # Allineato a LABELS[3] = "SUDDEN_DROP"

        # ── EVENTO 1: DERIVA LENTA DELLA TESTA (SLOW DRIFT) ───────────────────
        if delta > self.slow_drift_angle_thresh and abs_omega < self.slow_drift_max_gyro:
            if not self.in_drift:
                self.in_drift = True
                self.drift_start_time = t_now
            elif (t_now - self.drift_start_time) >= 1.5:
                log_system(f"🚨 [SLOW DRIFT] Testa inclinata in avanti stabilmente! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                self.last_event_time = t_now
                self.in_drift = False
                return 1  # Allineato a LABELS[1] = "SLOW_DRIFT"
        else:
            self.in_drift = False

        return 0
