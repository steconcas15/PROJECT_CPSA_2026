# event_dispatcher.py
# Dispatches recognized sensor events to the activation policy
# and triggers actions through the ActuatorManager

import queue
import threading
import time

from utils.event_queue import get_event_queue
from utils.logger import log_system, log_event

LABELS = {
    0: "AWAKE",                  # No drowsiness
    1: "SUSPECT_DROWSINESS",     # BlueCoin detects suspect head movement
    2: "CONFIRMED_DROWSINESS",   # Yolo+ResNet confirm drowsiness
}

ACTUATION_COOLDOWN = 5
# When YOLO is reacquiring, only switch back to MoveNet if the latest YOLO bbox is at most YOLO_REFRESH_MAX_AGE_SEC old.
YOLO_REFRESH_MAX_AGE_SEC = 1.0
# YOLO_REACQUIRE_MIN_INTERVAL_SEC: Do not request YOLO reacquisition more often than once every YOLO_REACQUIRE_MIN_INTERVAL_SEC 
YOLO_REACQUIRE_MIN_INTERVAL_SEC = 2.0
# YOLO_REACQUIRE_TIMEOUT_SEC: If YOLO cannot reacquire a person after YOLO_REACQUIRE_TIMEOUT_SEC, stop the video stage.
YOLO_REACQUIRE_TIMEOUT_SEC = 5.0

class EventDispatcher:
    """
    Consumes IMU & Video events and dispatches actions via activation policy.

    Video behavior:
        - tag=1 (Suspect) or tag=2 (Confirmed): Activate Video Pipeline.
        - YOLO runs first to find the driver's face.
        - Once the face is found, YOLO stops and ResNet starts classifying drowsiness.
        - tag=0 (Awake): Stop all video threads.
    """

    def __init__(
            self,
            actuator_manager,
            policy,
            yolo_thread=None,
            resnet_thread=None,
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
        self.resnet_thread = resnet_thread

        # None | "yolo_active" | "resnet_active"
        self._video_stage = None

        self._last_yolo_reacquire_request_ts = None
        self._yolo_started_ts = None

    def start(self):
        self._thread.start()
        log_system("[Dispatcher] Started.")

    def stop(self):
        self._stop_event.set()
        self._stop_video_threads()

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

    def _stop_video_threads(self):
        if self.yolo_thread and self.yolo_thread.is_active():
            log_system("[Dispatcher] Stopping YOLO thread.", level="INFO")
            self.yolo_thread.deactivate()
            self._wait_thread_idle(self.yolo_thread, "YOLO")

        if self.resnet_thread and self.resnet_thread.is_active():
            log_system("[Dispatcher] Stopping ResNet thread.", level="INFO")
            self.resnet_thread.deactivate()
            self._wait_thread_idle(self.resnet_thread, "ResNet")

        self._video_stage = None
        self._yolo_started_ts = None

        if self.roi_state is not None:
            self.roi_state.clear()

    def _activate_yolo(self, stage):
        if self.resnet_thread and self.resnet_thread.is_active():
            log_system("[Dispatcher] Stopping ResNet before activating YOLO.", level="INFO")
            self.resnet_thread.deactivate()
            self._wait_thread_idle(self.resnet_thread, "ResNet")

        if self.yolo_thread is None:
            log_system("[Dispatcher] YOLO thread not configured.", level="WARNING")
            self._video_stage = None
            return

        previous_stage = self._video_stage

        if not self.yolo_thread.is_active():
            log_system(f"[Dispatcher] Activating YOLO. stage={stage}", level="INFO")
            self.yolo_thread.activate()

        self._video_stage = stage

        if previous_stage != stage:
            self._yolo_started_ts = time.monotonic()

    def _activate_resnet(self, stage):
        if self.yolo_thread and self.yolo_thread.is_active():
            log_system("[Dispatcher] Stopping YOLO before activating ResNet.", level="INFO")
            self.yolo_thread.deactivate()
            self._wait_thread_idle(self.yolo_thread, "YOLO")

        if self.resnet_thread is None:
            log_system("[Dispatcher] ResNet thread not configured.", level="WARNING")
            self._video_stage = None
            return

        if not self.resnet_thread.is_active():
            log_system(f"[Dispatcher] Activating ResNet. stage={stage}", level="INFO")
            self.resnet_thread.activate()

        self._video_stage = stage

        self._yolo_started_ts = None

    def _apply_video_state_for_tag(self, tag):
        """
        Camera logic: 
        If tag is 1 (Suspect) or 2 (Confirmed), camera must be on.
        YOLO finds the face -> ResNet confirms drowsiness.
        """

        if tag in (1,2):
            if self._video_stage is None:
                self._activate_yolo()
            return

            # If YOLO is active, look for the person/face bounding box
            if self._video_stage == "yolo_active":
                if self.yolo_thread is None:
                    return

                person_bbox, person_conf, person_bbox_ts = (
                    self.yolo_thread.get_latest_person_bbox()
                )
                bbox_is_fresh = (
                    person_bbox is not None
                    and person_bbox_ts is not None
                    and (time.monotonic() - person_bbox_ts) <= YOLO_REFRESH_MAX_AGE_SEC
                )

                if bbox_is_fresh:
                    log_system(
                        f"[Dispatcher] Fresh YOLO person ROI acquired for tag=2. "
                        f"bbox={person_bbox}, conf={person_conf:.2f}. "
                        f"Switching to ResNet.",
                        level="INFO",
                    )
                    self._activate_resnet()

                    return

                # Timeout if YOLO cannot find the driver
                if self._yolo_started_ts is not None:
                    yolo_elapsed = time.monotonic() - self._yolo_started_ts

                    if yolo_elapsed >= YOLO_REACQUIRE_TIMEOUT_SEC:
                        log_system(
                            f"[Dispatcher] YOLO reacquisition timed out after "
                            f"{yolo_elapsed:.1f}s. Stopping video threads.",
                            level="WARNING",
                        )
                        self._stop_video_threads()

                return

            # If ResNet is active, check if tracking is lost (roi_state)
            if self._video_stage == "resnet_active":
                if self.resnet_thread is None:
                    return

                if self.roi_state is not None and self.roi_state.needs_reacquire():
                    now = time.monotonic()

                    if (
                        self._last_yolo_reacquire_request_ts is not None
                        and now - self._last_yolo_reacquire_request_ts < YOLO_REACQUIRE_MIN_INTERVAL_SEC
                    ):
                        return

                    self._last_yolo_reacquire_request_ts = now

                    log_system(
                        "[Dispatcher] ResNet requested ROI reacquisition. Switching back to YOLO.",
                        level="INFO",
                    )
                    self._activate_yolo()
                    return

                if not self.resnet_thread.is_active():
                    self._activate_resnet()

                return
        # If tag is 0 (Awake), turn off the camera to save resources
        self._stop_video_threads()

    def _trigger_policy_action(self, event):
        log_system(
            f"[Dispatcher POLICY IN] tag={event.get('drowsiness_tag')} "
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

        # Pass parameters to the actuator manager using the new action_type
        action_type = result["params"].get("action_type", "drowsiness_event")
        
        self.actuator_manager.trigger(
            actuator_id=result["actuator_id"],
            action_type=action_type,
            **result["params"],
        )

        return result

    def _should_trigger_policy(self, tag, now_time):
       # ACTUATION TRIGGER: happens ONLY if drowsiness is confirmed (tag == 2)
        if tag != 2:
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

                continue

            try:
                raw_tag = event.get("drowsiness_tag", "")

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
                    f"video_stage={self._video_stage}, "
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
                        source=event.get("source", "system"),
                    )

            except Exception as e:
                log_system(f"[Dispatcher] Dispatch error: {e}", level="ERROR")

            finally:
                try:
                    q.task_done()
                except Exception:
                    pass
