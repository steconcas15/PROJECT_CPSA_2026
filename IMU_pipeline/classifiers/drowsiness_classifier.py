# IMU_pipeline/classifiers/drowsiness_classifier.py
# Online streaming classifier for drowsiness detection via complementary filter.
# Aligned to process sample-by-sample subwindows streaming from the DataBuffer.

import numpy as np
import collections
from utils.logger import log_system

class DrowsinessClassifier:
    """
    Realtime adaptation of the drowsiness detection pipeline.
    Maintains cross-window filtering state and executes pattern logic for:
      1. SUDDEN DROP
      2. NOD
      3. SLOW DRIFT
    """
    def __init__(self, event_queue=None):
        self.event_queue = event_queue  # Coda per inviare eventi al dispatcher
        # Parametri di Tuning (allineati a drowsiness_detection.py)
        self.alpha = 0.96
        self.gate_thresh_g = 0.15
        self.target_fs = 100.0  # ODR stimata/target
        
        # Soglie eventi
        self.sudden_drop_gyro_thresh = 10.0
        self.sudden_drop_angle_thresh = 10.0
        self.nod_angle_min = 8.0
        self.nod_onset_gyro_thresh = 8.0
        self.slow_drift_angle_thresh = 10.0
        self.slow_drift_max_gyro = 12.0
        
        # Stati del filtro complementare e calibrazione
        self.theta_prev = 0.0
        self.baseline = 0.0
        self._calibrated = False
        
        # Buffer storici circolari (gestiscono fino a 30 secondi di dati a 100Hz per stima baseline)
        max_history_samples = int(30.0 * self.target_fs)
        self.theta_history = collections.deque(maxlen=max_history_samples)
        self.gyro_history = collections.deque(maxlen=max_history_samples)
        self.time_history = collections.deque(maxlen=max_history_samples)
        
        # Stati macchina a stati degli eventi
        self.in_nod = False
        self.nod_start_idx = 0
        self.nod_start_time = 0.0
        
        self.in_drift = False
        self.drift_start_time = 0.0
        
        self.last_baseline_update_time = 0.0
        self.last_event_time = -999.0

        # Flag interno per discriminare i nuovi campioni dagli overlap del buffer
        self.last_processed_ts = 0.0

        log_system("[DrowsinessClassifier] Realtime pipeline initialized.")

    def recognize(self, window_data: dict, window_end_ts: float):
        """
        Interfaccia di sink chiamata dal DataBuffer ad ogni emissione di finestra.
        window_data contiene i vettori float32 interi dell'attuale finestra.
        """
        accX = window_data['accX']
        accZ = window_data['accZ']
        gyrX = window_data['gyrX']
        ts_array = window_data['ts_array']
        hop_size = window_data['hop_size']

        # Per evitare di rielaborare i campioni sovrapposti derivanti dall'overlap del sliding window,
        # elaboriamo solo gli ultimi 'hop_size' campioni, tranne alla primissima finestra.
        if self.last_processed_ts == 0.0:
            start_idx = 0
        else:
            start_idx = len(accX) - hop_size

        if start_idx < 0 or start_idx >= len(accX):
            return

        for i in range(start_idx, len(accX)):
            ap_g = accX[i]
            av_g = accZ[i]
            omega = gyrX[i]
            t_now = ts_array[i]

            # Calcolo automatico del dt dinamico basato sui timestamp reali
            if len(self.time_history) > 0:
                dt = t_now - self.time_history[-1]
                if dt <= 0 or dt > 0.1:  # Fallback di sicurezza se c'è un buco temporale
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

            # 2. Aggiornamento baseline (Eseguito una volta al secondo per risparmiare CPU)
            if t_now - self.last_baseline_update_time >= 1.0:
                self.last_baseline_update_time = t_now
                theta_list = list(self.theta_history)
                if len(theta_list) > 10:
                    close_mask = [abs(th - self.baseline) < 5.0 for th in theta_list]
                    if sum(close_mask) > len(theta_list) // 4:
                        self.baseline = float(np.mean([theta_list[idx] for idx in range(len(theta_list)) if close_mask[idx]]))

            # 3. Analisi della Macchina a Stati per il rilevamento eventi
            self._process_state_machine(theta, omega, t_now)

        self.last_processed_ts = window_end_ts

    def _process_state_machine(self, theta: float, omega: float, t_now: float):
        # Protezione periodo refrattario allarmi (evita spam ravvicinato)
        if t_now - self.last_event_time < 1.0:
            return

        delta = theta - self.baseline
        abs_omega = abs(omega)

        # ── EVENTO 1: CADUTA BRUSCA ──────────────────────────────────────────
        sudden_samples = int(0.40 * self.target_fs)
        if len(self.theta_history) >= sudden_samples:
            recent_theta = list(self.theta_history)[-sudden_samples:]
            recent_delta = [th - self.baseline for th in recent_theta]
            recent_gyro = [abs(g) for g in list(self.gyro_history)[-sudden_samples:]]

            if (delta > self.sudden_drop_angle_thresh 
                    and abs_omega > self.sudden_drop_gyro_thresh 
                    and max(recent_gyro) > self.sudden_drop_gyro_thresh 
                    and (max(recent_delta) - min(recent_delta)) > self.sudden_drop_angle_thresh):
                
                log_system(f"🚨 [SUDDEN DROP] Rilevata caduta brusca della testa! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                if self.event_queue is not None:
                    try:
                        # Inserisce una tupla con (ID_sensore, Tag_evento, Timestamp)
                        self.event_queue.put_nowait(("bc_right", 1, t_now))
                    except Exception:
                        pass # Gestisce coda piena senza crashare l'IMU
                
                self.last_event_time = t_now
                self.in_nod = False
                self.in_drift = False
                return

        # ── EVENTO 2: AMMICCO (NOD) ──────────────────────────────────────────
        if not self.in_nod and abs_omega > self.nod_onset_gyro_thresh:
            self.in_nod = True
            self.nod_start_idx = len(self.theta_history) - 1
            self.nod_start_time = t_now

        if self.in_nod:
            elapsed_time = t_now - self.nod_start_time
            if abs(delta) < (self.nod_angle_min / 2.0):
                if 0.20 <= elapsed_time <= 3.0:
                    idx_start = self.nod_start_idx
                    theta_slice = list(self.theta_history)[idx_start:]
                    deltas_slice = [th - self.baseline for th in theta_slice]
                    if deltas_slice:
                        peak_delta = max(deltas_slice) if max(deltas_slice) > abs(min(deltas_slice)) else min(deltas_slice)
                        max_gyro = max([abs(g) for g in list(self.gyro_history)[idx_start:]])
                        if abs(peak_delta) > self.nod_angle_min:
                            log_system(f"🚨 [NOD] Rilevato colpo di sonno / ammicco! t={t_now:.2f}s, Durata={elapsed_time:.2f}s, Δθ_picco={peak_delta:.1f}°", level="WARNING")
                            # INVIO AL DISPATCHER TRAMITE CODA
                            if self.event_queue is not None:
                                try:
                                    self.event_queue.put_nowait(("bc_right", 2, t_now))
                                except Exception:
                                    pass
                            
                            self.last_event_time = t_now
                self.in_nod = False
            elif elapsed_time > 3.0:
                self.in_nod = False

        # ── EVENTO 3: DERIVA LENTA ────────────────────────────────────────────
        if delta > self.slow_drift_angle_thresh and abs_omega < self.slow_drift_max_gyro:
            if not self.in_drift:
                self.in_drift = True
                self.drift_start_time = t_now
            elif (t_now - self.drift_start_time) >= 2.5:
                log_system(f"🚨 [SLOW DRIFT] Testa inclinata in avanti stabilmente (Deriva Lenta)! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                # INVIO AL DISPATCHER TRAMITE CODA
                if self.event_queue is not None:
                    try:
                        self.event_queue.put_nowait(("bc_right", 3, t_now))
                    except Exception:
                        pass
                
                self.last_event_time = t_now
                self.in_drift = False
        else:
            self.in_drift = False