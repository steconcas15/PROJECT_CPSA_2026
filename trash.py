# IMU_pipeline/classifiers/drowsiness_classifier.py
# Online streaming classifier for drowsiness detection via complementary filter.
# Aligned to process sample-by-sample subwindows streaming from the DataBuffer.

import numpy as np
import collections
import uuid
from datetime import datetime
from utils.logger import log_system, log_event
from utils.event_queue import enqueue_drop_oldest, get_event_queue  # Importati da stereotipy

class DrowsinessClassifier:
    """
    Realtime adaptation of the drowsiness detection pipeline.
    Maintains cross-window filtering state and executes pattern logic for:
      1. SUDDEN DROP
      2. NOD
      3. SLOW DRIFT
    """
    def __init__(self, event_queue=None):
        # Ora utilizziamo la coda globale di sistema come lo stereotipy_classifier,
        # mantenendo event_queue come fallback/opzionale se passato da fuori.
        self.q = event_queue if event_queue is not None else get_event_queue()
        self.source = "bc_right"  # Identificativo sorgente per drowsiness
        
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

    def _send_drowsiness_event(self, tag: int, t_now: float, extra_features: list = None):
        """
        Funzione helper privata per impacchettare l'evento nel dizionario standard
        e gestirne l'accodamento con drop dell'elemento più vecchio (Stile Stereotipy).
        """
        event = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now().isoformat(),
            "window_ts": float(t_now),
            "source": self.source,
            "features": extra_features if extra_features is not None else [],
            "stereotipy_tag": str(tag)  # Il dispatcher fa il cast int() su questa chiave
        }

        # Invio gestito con enqueue_drop_oldest per proteggere la coda da overflow
        dropped, dropped_item = enqueue_drop_oldest(self.q, event, kind="imu")
        if dropped:
            log_system("[DrowsinessClassifier] Oldest event dropped (queue full)", level="WARNING")
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

        self.last_event_time = t_now

    def _process_state_machine(self, theta: float, omega: float, t_now: float):
        # Protezione periodo refrattario allarmi (evita spam ravvicinato)
        if t_now - self.last_event_time < 1.0:
            return

        delta = theta - self.baseline
        abs_omega = abs(omega)

        # Mappatura LABELS del dispatcher: 1: SLOW_DRIFT, 2: NOD, 3: SUDDEN_DROP
        # ── EVENTO 3: CADUTA BRUSCA (SUDDEN_DROP) ──────────────────────────
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
                
                # ADATTAMENTO: Invio del dizionario anziché della tupla. Tag = 3 per SUDDEN_DROP
                self._send_drowsiness_event(tag=3, t_now=t_now, extra_features=[float(delta), float(abs_omega)])
                
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
                            
                            # ADATTAMENTO: Tag = 2 per NOD
                            self._send_drowsiness_event(tag=2, t_now=t_now, extra_features=[float(peak_delta), float(max_gyro)])
                            
                self.in_nod = False
            elif elapsed_time > 3.0:
                self.in_nod = False

        # ── EVENTO 1: DERIVA LENTA (SLOW_DRIFT) ────────────────────────────────
        if delta > self.slow_drift_angle_thresh and abs_omega < self.slow_drift_max_gyro:
            if not self.in_drift:
                self.in_drift = True
                self.drift_start_time = t_now
            elif (t_now - self.drift_start_time) >= 2.5:
                log_system(f"🚨 [SLOW DRIFT] Testa inclinata in avanti stabilmente (Deriva Lenta)! t={t_now:.2f}s, Δθ={delta:.1f}°", level="WARNING")
                
                # ADATTAMENTO: Tag = 1 per SLOW_DRIFT
                self._send_drowsiness_event(tag=1, t_now=t_now, extra_features=[float(delta), float(abs_omega)])
                
                self.in_drift = False
        else:
            self.in_drift = False