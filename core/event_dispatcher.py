# event_dispatcher.py
import queue
import threading
import time

from utils.event_queue import get_event_queue
from utils.logger import log_system, log_event

LABELS = {
    0: "SVEGLIO_OK",
    1: "SLOW_DRIFT",
    2: "NOD",
    3: "SUDDEN_DROP",
}

ACTUATION_COOLDOWN = 5
AWAKE_OFF_DELAY_SEC = 5.0

class EventDispatcher:
    def __init__(self, actuator_manager, policy, yolo_thread=None, roi_state=None):
        self.actuator_manager = actuator_manager
        self.policy = policy
        self.roi_state = roi_state
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._last_tag = None
        self._last_actuation_time = None
        self._latest_tag_lock = threading.Lock()
        self._latest_tag = None
        self.yolo_thread = yolo_thread
        self._video_on = False
        self._last_activate_attempt_ts = None
        self._activate_retry_interval_sec = 1.0
        self._awake_since_ts = None
        # Timer per calcolare la durata continuativa dello stato DROWSY
        self._drowsy_since_ts = None

    def start(self):
        self._thread.start()
        log_system("[Dispatcher] Started.")

    def stop(self):
        self._stop_event.set()
        self._stop_video_thread()
        if self._thread.is_alive():
            self._thread.join()
        log_system("[Dispatcher] Stopped.")

    def _set_latest_tag(self, tag, event):
        with self._latest_tag_lock:
            self._latest_tag = tag

    def _get_latest_tag(self):
        with self._latest_tag_lock:
            return self._latest_tag

    def _stop_video_thread(self):
        if self.yolo_thread and self.yolo_thread.is_active():
            self.yolo_thread.deactivate()
        self._video_on = False
        self._awake_since_ts = None
        self._drowsy_since_ts = None
        if self.roi_state: self.roi_state.clear()

    def _start_video_thread(self):
        if self.yolo_thread and not self.yolo_thread.is_active():
            now = time.monotonic()
            if self._last_activate_attempt_ts is None or (now - self._last_activate_attempt_ts) > self._activate_retry_interval_sec:
                self._last_activate_attempt_ts = now
                self.yolo_thread.activate()
                self._video_on = True

    def _evaluate_resnet_off_timer(self):
        if not self._video_on or not self.yolo_thread:
            self._awake_since_ts = None
            self._drowsy_since_ts = None
            return
        
        prediction = self.yolo_thread.get_latest_prediction()
        now = time.monotonic()

        # 1. Gestione DROWSY con filtro per i blink (soglia >= 1 secondo)
        if prediction == "DROWSY":
            if self._drowsy_since_ts is None:
                self._drowsy_since_ts = now
            
            # Se la sonnolenza persiste da più di 1 secondo, resettiamo il timer di spegnimento
            if (now - self._drowsy_since_ts) >= 1.0:
                self._awake_since_ts = None
            return
        else:
            # Se la predizione non è DROWSY, azzeriamo il timer della sonnolenza
            self._drowsy_since_ts = None

        # 2. Gestione NATURAL per il conteggio dello spegnimento camera
        if prediction == "NATURAL":
            if self._awake_since_ts is None: 
                self._awake_since_ts = now
            elif (now - self._awake_since_ts) >= AWAKE_OFF_DELAY_SEC:
                self._stop_video_thread()
            return

        # 3. Altri stati transienti o predizioni nulle (es. nessun volto rilevato)
        # Non resettano il timer di spegnimento, lasciando scorrere il tempo normalmente

    def _should_trigger_policy(self, tag, now_time):
        # Se il video è spento, serve il trigger IMU
        if not self._video_on:
            return tag in (1, 2, 3) and (self._last_actuation_time is None or (now_time - self._last_actuation_time) >= ACTUATION_COOLDOWN)
        
        # Se il video è acceso, ignoriamo l'IMU e guardiamo solo il cooldown
        return self._last_actuation_time is None or (now_time - self._last_actuation_time) >= ACTUATION_COOLDOWN

    def _process_policy_for_event(self, event, tag, now_time):
        if not self._should_trigger_policy(tag, now_time):
            return None
        
        current_pred = self.yolo_thread.get_latest_prediction() if self.yolo_thread else None
        result = self.policy.handle(event, video_prediction=current_pred)
        
        if result:
            self._last_actuation_time = now_time
            self.actuator_manager.trigger(result["actuator_id"], "stereotipy_event", **result["params"])
        return result

    def _process_events(self):
        q = get_event_queue()
        while not self._stop_event.is_set():
            try:
                event = q.get(timeout=0.5)
                tag = int(event.get("stereotipy_tag", 0))
                self._set_latest_tag(tag, event)
                self._apply_video_state_for_tag(tag)
                self._process_policy_for_event(event, tag, time.monotonic())
                q.task_done()
            except:
                if self._video_on:
                    self._evaluate_resnet_off_timer()
                    # Trigger continuo basato solo su ResNet se video_on
                    self._process_policy_for_event({}, None, time.monotonic())

    def _apply_video_state_for_tag(self, tag):
        if tag in (1, 2, 3): self._start_video_thread()
