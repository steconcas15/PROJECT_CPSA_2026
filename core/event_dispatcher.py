# event_dispatcher.py
# Dispatches recognized sensor events to the activation policy
# and triggers actions through the ActuatorManager.
#
# Video pipeline: YOLO+ResNet integrato (un unico thread, YoloDpuThread).
# MoveNet non fa piu' parte del progetto: rimosso ogni riferimento.

import queue
import threading
import time

from utils.event_queue import get_event_queue
from utils.logger import log_system, log_event

# Mappatura aggiornata per il caso Drowsiness (Sonnolenza)
LABELS = {
    0: "SVEGLIO_OK",
    1: "SLOW_DRIFT",    # Deriva lenta della testa
    2: "NOD",           # Colpo di sonno / oscillazione
    3: "SUDDEN_DROP",   # Crollo brusco della testa (Massimo pericolo)
}

ACTUATION_COOLDOWN = 5

# Quanto tempo ResNet deve riportare "NATURAL" ininterrottamente prima di
# spegnere la camera. Si resetta a ogni predizione ResNet "DROWSY".
AWAKE_OFF_DELAY_SEC = 5.0


class EventDispatcher:
    """
    Consumes IMU classifier events and dispatches actions via activation policy.

    Video behavior (Drowsiness):
        - For tag=1, 2, 3 (Any drowsiness sign from IMU): activates the
          YOLO+ResNet thread. This is a trigger only.
        - Deactivation is governed exclusively by ResNet's prediction on the
          video bbox (DROWSY/NATURAL), NOT by the IMU tag: the camera turns
          off only after ResNet reports NATURAL for AWAKE_OFF_DELAY_SEC
          seconds uninterrupted. IMU and ResNet are independent sensors;
          the head (IMU) can return to neutral while the face/posture seen
          on camera (ResNet) still shows drowsiness, so the IMU tag alone
          must never turn the camera off.

    The dispatcher is the sole authority on when the video thread should be
    active. The video thread itself never self-deactivates; this avoids the
    state desync that caused the camera to stay off after a single failure.
    """

    def __init__(
            self,
            actuator_manager,
            policy,
            yolo_thread=None,
            roi_state=None,
        ):
        self.actuator_manager = actuator_manager
        self.policy = policy
        self.roi_state = roi_state

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_events, daemon=True)

        self._last_tag = None
        self._last_actuation_time = None

        self._latest_tag_lock = threading.Lock()
        self._latest_tag = None
        self._latest_event = None

        self.yolo_thread = yolo_thread

        # Stato video: False (off) o True (on). Niente piu' stage intermedi.
        self._video_on = False

        # Throttle per non spammare activate() se il thread fatica ad aprire la camera.
        self._last_activate_attempt_ts = None
        self._activate_retry_interval_sec = 1.0

        # Timer di spegnimento: parte quando ResNet riporta NATURAL e si
        # azzera ogni volta che ResNet riporta DROWSY. La camera si spegne
        # solo se ResNet resta "NATURAL" ininterrottamente per
        # AWAKE_OFF_DELAY_SEC secondi. Governato esclusivamente da ResNet,
        # NON dal tag IMU (vedi _evaluate_resnet_off_timer).
        self._awake_since_ts = None

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
            self._latest_event = event

    def _get_latest_tag(self):
        with self._latest_tag_lock:
            return self._latest_tag

    def _wait_thread_idle(self, thread, name, timeout_sec=5.0):
        if thread is None:
            return True

        start = time.monotonic()

        while not self._stop_event.is_set():
            if getattr(thread, "phase", None) == "idle":
                return True

            if time.monotonic() - start >= timeout_sec:
                log_system(
                    f"[Dispatcher] Timeout waiting for {name} to become idle. "
                    f"Current phase={getattr(thread, 'phase', 'unknown')}",
                    level="WARNING",
                )
                return False

            time.sleep(0.05)

        return False

    def _stop_video_thread(self):
        if self.yolo_thread and self.yolo_thread.is_active():
            log_system("[Dispatcher] Stopping YOLO+ResNet thread.", level="INFO")
            self.yolo_thread.deactivate()
            self._wait_thread_idle(self.yolo_thread, "YOLO")

        self._video_on = False
        self._last_activate_attempt_ts = None
        self._awake_since_ts = None

        if self.roi_state is not None:
            self.roi_state.clear()

    def _start_video_thread(self):
        if self.yolo_thread is None:
            log_system("[Dispatcher] YOLO thread not configured.", level="WARNING")
            return

        # Se il thread e' gia' attivo (active_event set), non richiamiamo activate()
        # di nuovo: e' idempotente ma evitiamo log/rumore inutile.
        if self.yolo_thread.is_active():
            self._video_on = True
            return

        now = time.monotonic()
        if (
            self._last_activate_attempt_ts is not None
            and (now - self._last_activate_attempt_ts) < self._activate_retry_interval_sec
        ):
            # Throttle: non richiamare activate() ad ogni singolo poll/evento.
            # is_active() resta False finche' il thread non apre la camera,
            # quindi senza throttle qui rientreremmo in questo branch di
            # continuo finche' la camera non si apre.
            return

        self._last_activate_attempt_ts = now

        log_system("[Dispatcher] Activating YOLO+ResNet thread.", level="INFO")
        self.yolo_thread.activate()
        self._video_on = True

    def _apply_video_state_for_tag(self, tag):
        """
        L'IMU (tag 1/2/3) e' solo il TRIGGER di accensione della camera.
        Una volta accesa, lo spegnimento NON dipende piu' dal tag IMU:
        e' deciso esclusivamente da ResNet (vedi _evaluate_resnet_off_timer),
        che guarda la predizione sulla bbox video frame per frame.

        Questo perche' IMU e ResNet sono due sensori indipendenti: la testa
        (IMU) puo' tornare in posizione neutra mentre il volto/postura
        ripresi dalla camera (ResNet) mostrano ancora sonnolenza, e in quel
        caso la camera deve restare accesa finche' ResNet non confirma
        "sveglio" per AWAKE_OFF_DELAY_SEC secondi ininterrotti.
        """
        if tag in (1, 2, 3):
            self._start_video_thread()
            return

        # tag == 0 o non valido: non spegne nulla qui. Lo spegnimento e'
        # competenza esclusiva di _evaluate_resnet_off_timer().

    def _evaluate_resnet_off_timer(self):
        """
        Timer di spegnimento basato SOLO sulla predizione ResNet:

          - prediction == "DROWSY": azzera il timer (qualsiasi segno di
            sonnolenza visivo resetta il conto alla rovescia).
          - prediction == "NATURAL": fa partire (o continua) il timer;
            la camera si spegne solo se resta "NATURAL" ininterrottamente
            per AWAKE_OFF_DELAY_SEC secondi.
          - prediction is None (camera spenta, o nessun frame ancora
            elaborato dopo l'accensione): non fa partire nulla.

        Va chiamato periodicamente (poll), non solo agli eventi IMU,
        perche' ResNet produce un nuovo risultato per ogni frame video,
        a un ritmo indipendente dal flusso di eventi del classificatore IMU.
        """
        if not self._video_on or self.yolo_thread is None:
            self._awake_since_ts = None
            return

        prediction = self.yolo_thread.get_latest_prediction()

        if prediction == "DROWSY":
            self._awake_since_ts = None
            return

        if prediction != "NATURAL":
            # None (nessun frame valido ancora, es. subito dopo l'accensione
            # o nessuna persona rilevata): non avviamo il conteggio.
            return

        now = time.monotonic()

        if self._awake_since_ts is None:
            self._awake_since_ts = now
            return

        if (now - self._awake_since_ts) >= AWAKE_OFF_DELAY_SEC:
            log_system(
                f"[Dispatcher] ResNet ha rilevato NATURAL per "
                f"{AWAKE_OFF_DELAY_SEC:.0f}s ininterrotti. Spegnimento camera.",
                level="INFO",
            )
            self._stop_video_thread()
            self._awake_since_ts = None

    def _trigger_policy_action(self, event):
        log_system(
            f"[Dispatcher POLICY IN] tag={event.get('stereotipy_tag')} "
            f"source={event.get('source')}",
            level="INFO",
        )

        result = self.policy.handle(event)

        log_system(
            f"[Dispatcher POLICY OUT] result={result}",
            level="INFO",
        )

        if not result:
            log_system("[Dispatcher] Policy returned no action.")
            return None

        self.actuator_manager.trigger(
            actuator_id=result["actuator_id"],
            action_type="stereotipy_event",
            **result["params"],
        )

        return result

    def _should_trigger_policy(self, tag, now_time):
        if tag not in (1, 2, 3):
            return False

        return (
            self._last_actuation_time is None
            or (now_time - self._last_actuation_time) >= ACTUATION_COOLDOWN
        )

    def _process_policy_for_event(self, event, tag, now_time):
        if not self._should_trigger_policy(tag, now_time):
            return None

        try:
            result = self._trigger_policy_action(event)
        except Exception as e:
            log_system(f"[Dispatcher] Trigger error: {e}", level="ERROR")
            return None

        if result:
            self._last_actuation_time = now_time

        return result

    def _process_events(self):
        q = get_event_queue()

        while not self._stop_event.is_set():
            try:
                event = q.get(timeout=0.5)
            except queue.Empty:
                latest_tag = self._get_latest_tag()

                if latest_tag is not None:
                    self._apply_video_state_for_tag(latest_tag)

                # ResNet detta lo spegnimento indipendentemente dal flusso
                # di eventi IMU: va valutato ad ogni giro, coda vuota o no.
                self._evaluate_resnet_off_timer()
                continue

            try:
                raw_tag = event.get("stereotipy_tag", "")

                try:
                    tag = int(raw_tag)
                except Exception:
                    tag = None

                label = LABELS.get(tag, str(raw_tag))
                now_time = time.monotonic()

                self._set_latest_tag(tag, event)

                log_system(
                    f"[Dispatcher IN] raw_tag={raw_tag}, tag={tag}, "
                    f"label={label}, last_tag={self._last_tag}, "
                    f"video_on={self._video_on}, "
                    f"resnet_pred={self.yolo_thread.get_latest_prediction() if self.yolo_thread else None}, "
                    f"queue_size={q.qsize() if hasattr(q, 'qsize') else 'unknown'}",
                    level="INFO",
                )

                tag_changed = tag != self._last_tag

                if tag_changed:
                    log_system(
                        f"[Dispatcher] Tag changed: {self._last_tag} -> {tag} ({label})",
                        level="INFO",
                    )

                    self._last_tag = tag
                    self._last_actuation_time = None

                self._apply_video_state_for_tag(tag)
                self._evaluate_resnet_off_timer()

                result = self._process_policy_for_event(event, tag, now_time)

                actuations = []
                if result:
                    actuations = [
                        {
                            "target": result["actuator_id"],
                            "params": result["params"],
                        }
                    ]

                if tag_changed:
                    log_event(
                        timestamp=event.get("timestamp"),
                        feature_type="imu",
                        event=label,
                        actuations=actuations,
                        source=event.get("source", "dual_wrist"),
                    )

            except Exception as e:
                log_system(f"[Dispatcher] Dispatch error: {e}", level="ERROR")

            finally:
                try:
                    q.task_done()
                except Exception:
                    pass
