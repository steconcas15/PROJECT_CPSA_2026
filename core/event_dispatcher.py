# event_dispatcher.py
import queue
import threading
import time

from utils.event_queue import get_event_queue
from utils.logger import log_system, log_event

# Mapping of numerical tags from sensors (e.g., IMU) to human-readable labels
LABELS = {
    0: "AWAKE",
    1: "SLOW_DRIFT",
    2: "SUDDEN_DROP",
}

# Cooldown (in seconds) to prevent repeated actuator triggers within a short window
ACTUATION_COOLDOWN = 5

# Delay (in seconds) in NATURAL state before shutting down the camera
AWAKE_OFF_DELAY_SEC = 5.0

class EventDispatcher:
    """
    Handles system event dispatching and flow control.
    Coordinates sensor data (IMU) and video feedback (YOLO/ResNet) to trigger
    appropriate countermeasures (policies) via actuators, while managing camera states.
    """
    def __init__(self, actuator_manager, policy, yolo_thread=None, roi_state=None):
        self.actuator_manager = actuator_manager
        self.policy = policy
        self.roi_state = roi_state

        # Main Dispatcher thread management
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_events, daemon=True)

        # Actuator activation state tracking
        self._last_tag = None
        self._last_actuation_time = None

        # Thread safe synchronization for the latest received tag
        self._latest_tag_lock = threading.Lock()
        self._latest_tag = None

        # Video pipeline state
        self.yolo_thread = yolo_thread
        self._video_on = False
        self._last_activate_attempt_ts = None
        self._activate_retry_interval_sec = 1.0

        # Timestamps for tracking durations
        self._awake_since_ts = None    # How long the user has been detected as awake (NATURAL)
        self._drowsy_since_ts = None   # How long the user has been continuously drowsy (DROWSY)

    def start(self):
        """
        Starts the background thread for event processing.
        """ 
        self._thread.start()
        log_system("[Dispatcher] Started.")

    def stop(self):
        """
        Stops the dispatcher thread and safely shuts down video modules.
        """
        self._stop_event.set()
        self._stop_video_thread()
        if self._thread.is_alive():
            self._thread.join()
        log_system("[Dispatcher] Stopped.")

    def _set_latest_tag(self, tag, event):
        """
        Updates the latest tag in a thread-safe manner.
        """
        with self._latest_tag_lock:
            self._latest_tag = tag

    def _get_latest_tag(self):
        """
        Retrieves the latest registered tag in a thread safe manner.
        """
        with self._latest_tag_lock:
            return self._latest_tag

    def _stop_video_thread(self):
        """
        Shuts down the video processing thread and resets related state timers.
        """
        if self.yolo_thread and self.yolo_thread.is_active():
            self.yolo_thread.deactivate()
        self._video_on = False
        self._awake_since_ts = None
        self._drowsy_since_ts = None
        if self.roi_state: self.roi_state.clear()

    def _start_video_thread(self):
        """
        Activates the video thread (YOLO/ResNet)
        """
        if self.yolo_thread and not self.yolo_thread.is_active():
            now = time.monotonic()
            if self._last_activate_attempt_ts is None or (now - self._last_activate_attempt_ts) > self._activate_retry_interval_sec:
                self._last_activate_attempt_ts = now
                self.yolo_thread.activate()
                self._video_on = True

    def _evaluate_resnet_off_timer(self):
        """
        Analyzes video model predictions to manage camera shutdown
        and filter out false positives caused by normal eye blinking.
        """
        if not self._video_on or not self.yolo_thread:
            self._awake_since_ts = None
            self._drowsy_since_ts = None
            return
        
        prediction = self.yolo_thread.get_latest_prediction()
        now = time.monotonic()

        # 1. DROWSY MANAGEMENT: Anti-blink filter (Threshold >= 1.0 second)
        # If the user closes their eyes just for a normal blink, we shouldn't block the shutdown timer.
        
        if prediction == "DROWSY":
            if self._drowsy_since_ts is None:
                self._drowsy_since_ts = now
            
            # If drowsiness persists continuously for at least 1 second, it's a real microsleep.
            # In this case, reset the camera shutdown timer (the camera must stay on).
            if (now - self._drowsy_since_ts) >= 1.0:
                self._awake_since_ts = None
            return
        else:
            # If the state is no longer DROWSY, immediately reset the drowsiness timer
            self._drowsy_since_ts = None

        # 2. NATURAL MANAGEMENT: Camera shutdown countdown
        # If the user is alert/awake, start the countdown to turn off the camera and save resources.
        if prediction == "NATURAL":
            if self._awake_since_ts is None: 
                self._awake_since_ts = now
            elif (now - self._awake_since_ts) >= AWAKE_OFF_DELAY_SEC:
                # If the NATURAL state persists longer than AWAKE_OFF_DELAY_SEC, turn off the video
                self._stop_video_thread()
            return

        # 3. TRANSIENT STATES (e.g., no face detected or null predictions)
        # These states do not reset the timers, allowing the countdowns to tick away uninterrupted.

    def _should_trigger_policy(self, tag, now_time):
        """
        Determines whether current conditions allow executing a countermeasure (policy).
        Handles sensor priority and actuator cooldown rules.
        """
        # Scenario A: If the video is OFF, rely entirely on the IMU tags.
        # Trigger only if the tag indicates an anomaly (1 or 3) and the cooldown has elapsed.
        if not self._video_on:
            return tag in (1, 2) and (self._last_actuation_time is None or (now_time - self._last_actuation_time) >= ACTUATION_COOLDOWN)
        
        # Scenario B: If the video is ON, ignore raw IMU tags (the video prediction takes precedence).
        # Check exclusively whether the actuator cooldown window has passed.
        return self._last_actuation_time is None or (now_time - self._last_actuation_time) >= ACTUATION_COOLDOWN

    def _process_policy_for_event(self, event, tag, now_time):
        """
        Evaluates the current event against active policies and commands 
        physical actuators (e.g., speaker) if necessary.
        """
        # Preliminary check to see if a policy trigger is allowed (cooldown or video state)
        if not self._should_trigger_policy(tag, now_time):
            return None
        
        current_pred = self.yolo_thread.get_latest_prediction() if self.yolo_thread else None
        
        # --- SPEAKER ANTI-BLINK FILTER ---
        # Prevents the speaker from firing due to a physiological eye blink (duration < 1s)
        if current_pred == "DROWSY":
            # If the timer hasn't started or a full continuous second of drowsiness hasn't elapsed yet
            if self._drowsy_since_ts is None or (now_time - self._drowsy_since_ts) < 1.0:
                current_pred = None  # Temporarily override to None to suppress the audio alarm

        # Query the policy module to get the required action based on the event and filtered video prediction
        result = self.policy.handle(event, video_prediction=current_pred)

        # If the policy yields a valid action, trigger the corresponding actuator
        if result:
            self._last_actuation_time = now_time
            self.actuator_manager.trigger(result["actuator_id"], "drowsiness_event", **result["params"])
        return result

    def _process_events(self):
        """
        Main thread loop. Continuously fetches events from the queue,
        updates timers on timeouts, and monitors overall system state.
        """
        q = get_event_queue()
        while not self._stop_event.is_set():
            try:
                # Wait for an event from the queue (0.5s timeout prevents blocking the thread shutdown)
                event = q.get(timeout=0.5)
                tag = int(event.get("drowsiness_tag", 0))
                self._set_latest_tag(tag, event)
                self._apply_video_state_for_tag(tag) # Turn video on if the IMU detects anomalies
                self._process_policy_for_event(event, tag, time.monotonic())
                q.task_done()
            except:
                # If the queue is empty (0.5s timeout reached), the loop doesn't block:
                # it takes the opportunity to handle shutdown timers and evaluate pure video-driven policies.
                if self._video_on:
                    self._evaluate_resnet_off_timer()
                    # Execute continuous policy assessment using only the current video prediction
                    self._process_policy_for_event({}, None, time.monotonic())

    def _apply_video_state_for_tag(self, tag):
        """ 
        Activates the video pipeline if IMU sensors detect anomalies (SLOW_DRIFT or SUDDEN_DROP).
        """
        if tag in (1, 2): self._start_video_thread()
