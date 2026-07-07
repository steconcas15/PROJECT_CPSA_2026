# sensor_manager.py
#
# Author: Stefano Concas, Matteo Matta, Sara Cadedo, Fabio Piras
# Repository: https://github.com/steconcas15/PROJECT_CPSA_2026

from blue_st_sdk.features.feature_accelerometer import FeatureAccelerometer
from blue_st_sdk.features.feature_gyroscope import FeatureGyroscope

from sensors.BLE.bluecoin import scan_bluecoin_devices, BlueCoinThread
from sensors.BLE.feature_listeners import (
    AccelerometerFeatureListener,
    GyroscopeFeatureListener,
)

from IMU_pipeline.data_stream.synchronizer import IMUSynchronizer
from IMU_pipeline.classifiers.drowsiness_classifier import DrowsinessClassifier

from utils.config import get_bluecoin_config
from utils.logger import log_system
from utils.lock import device_scan_lock, device_connection_lock


class SensorManager:
    """
    Owns and coordinates all hardware-facing objects for a single BlueCoin device.

    Responsibilities:
      1. BLE discovery (scan_sensors)
      2. Pipeline wiring: classifier sink registered on the buffer (in __init__)
      3. Device connection and feature listener registration (initialize_sensors)
      4. Graceful shutdown of BLE threads (stop_all)

    Lifecycle expected by main.py:
      sm = SensorManager()          # builds pipeline objects, wires sink
      sm.scan_sensors()             # BLE scan — may be retried by main.py
      sm.initialize_sensors()       # connects device, starts BLE thread
      ...                           # system runs
      sm.stop_all()                 # shutdown
    """

    def __init__(self):
        # --- Internal state --------------------------------------------------
        self.threads   = []    # list of active BlueCoinThread instances
        self.bluecoins = []    # BlueST Node objects found during the last scan

        # Load the list of expected devices from config.yaml (bluecoins section).
        # Used in initialize_sensors() to look up the target node by logical id.
        self.config = get_bluecoin_config()

        # --- Pipeline construction -------------------------------------------
        # IMUSynchronizer collects acc and gyr updates, pairs them by timestamp,
        # and forwards aligned rows to its internal DataBuffer.
        self.synchronizer = IMUSynchronizer()

        # DrowsinessClassifier is the sliding-window sink: it receives one
        # structured window dict per hop and produces events on the global queue.
        self.classifier = DrowsinessClassifier()

        # Wire the classifier as the buffer's consumer.
        # From this point forward, every completed window emitted by DataBuffer
        # will call classifier.recognize(window_payload, window_end_ts).
        self.synchronizer.buffer.set_features_sink(self.classifier.recognize)

        log_system("[SensorManager] Initialized for single-device Drowsiness Detection pipeline")


    # =========================================================================
    #  PUBLIC INTERFACE
    # =========================================================================

    def scan_sensors(self):
        """
        Perform a BLE scan and store discovered BlueCoin nodes in self.bluecoins.

        Uses device_scan_lock to prevent concurrent scans if this method is
        called from multiple threads or retried rapidly by main.py.
        After the scan, self.bluecoins is replaced entirely — any previous
        results are discarded.
        """
        log_system("[SensorManager] Starting BLE scan for BlueCoin devices")

        with device_scan_lock:
            self.bluecoins = scan_bluecoin_devices(timeout=5)

        log_system(f"[SensorManager] Found {len(self.bluecoins)} BlueCoin device(s)")

        # Log discovered names for diagnostics; errors here are non-fatal
        # (a node with an unreadable name is still stored in self.bluecoins).
        for node in self.bluecoins:
            try:
                log_system(f"[SensorManager] Discovered: {node.get_name()}")
            except Exception as e:
                log_system(f"[SensorManager] Error retrieving name for scanned node: {e}",
                           level="WARNING")

    def initialize_sensors(self):
        """
        Connect to the bc_left BlueCoin and start its acquisition thread.

        Looks up the target node by the name declared in config.yaml under
        bluecoins[id=bc_left], retrieves Accelerometer and Gyroscope features,
        attaches the corresponding listeners, and starts a BlueCoinThread.

        The thread runs the full BLE lifecycle (connect → notify → reconnect on
        drop) independently from the main thread. Listeners inside the thread
        call synchronizer.update() on every incoming BLE packet.

        Early returns with an ERROR log on any unrecoverable configuration or
        hardware issue; does not raise exceptions.
        """
        if not self.bluecoins:
            log_system(
                "[SensorManager] No scanned bluecoins available. "
                "Run scan_sensors() first.",
                level="WARNING"
            )
            return

        # --- Build name → node lookup map ------------------------------------
        by_name = {}
        for node in self.bluecoins:
            try:
                name = node.get_name()
                if name:
                    by_name[name] = node
            except Exception as e:
                log_system(f"[SensorManager] Can't read node name: {e}", level="WARNING")

        # --- Resolve target node from config ---------------------------------
        # config.yaml declares the device as:
        #   bluecoins:
        #     - id:   bc_left
        #       name: "CPSA_L2"
        # We resolve "bc_left" → the BLE advertisement name → the Node object.
        expected  = {c.get("id"): c.get("name")
                     for c in self.config if c.get("id") and c.get("name")}
        left_name = expected.get("bc_left")

        if not left_name:
            log_system(
                "[SensorManager] Config must include bc_left with a matching name.",
                level="ERROR"
            )
            return

        if left_name not in by_name:
            log_system(
                f"[SensorManager] Missing expected left BlueCoin device: '{left_name}'. "
                f"Discovered nodes: {list(by_name.keys())}",
                level="ERROR"
            )
            return

        sensor_id     = "bc_left"         # logical id used by the synchronizer
        expected_name = left_name         # BLE advertisement name
        node          = by_name[expected_name]

        # --- Retrieve BLE feature objects ------------------------------------
        # Features are BlueST SDK objects that wrap BLE characteristic handles.
        # Getting a feature does not enable notifications yet — that happens
        # inside BlueCoinThread._start_notifications().
        try:
            feat_acc = node.get_feature(FeatureAccelerometer)
            feat_gyr = node.get_feature(FeatureGyroscope)
        except Exception as e:
            log_system(
                f"[SensorManager] Error retrieving features for '{expected_name}': {e}",
                level="ERROR"
            )
            return

        # --- Build feature + listener pairs ----------------------------------
        # Each listener is bound to this sensor_id so that synchronizer.update()
        # can route the data to the correct _DevState slot.
        features, listeners = [], []

        if feat_acc:
            features.append(feat_acc)
            listeners.append(
                AccelerometerFeatureListener(
                    device_id=sensor_id, synchronizer=self.synchronizer
                )
            )
        else:
            log_system(f"[SensorManager] {expected_name} is missing Accelerometer",
                       level="WARNING")

        if feat_gyr:
            features.append(feat_gyr)
            listeners.append(
                GyroscopeFeatureListener(
                    device_id=sensor_id, synchronizer=self.synchronizer
                )
            )
        else:
            log_system(f"[SensorManager] {expected_name} is missing Gyroscope",
                       level="WARNING")

        if not features:
            # Both features missing — nothing to stream, abort.
            log_system(
                f"[SensorManager] No target features available on node {expected_name}",
                level="WARNING"
            )
            return

        # --- Start the BLE acquisition thread --------------------------------
        # device_connection_lock serialises connect() calls; the BlueST SDK
        # is not safe for concurrent BLE connections from multiple threads.
        try:
            with device_connection_lock:
                thread = BlueCoinThread(
                    node             = node,
                    feature          = features,
                    feature_listener = listeners,
                    device_id        = sensor_id,
                )
                thread.start()          # begins _connect() → _listen() lifecycle
                self.threads.append(thread)

            log_system(
                f"[SensorManager] Sensor initialized: {sensor_id} ({expected_name}) "
                f"with {len(features)} active features"
            )
        except Exception as e:
            log_system(
                f"[SensorManager] Error initializing thread for '{sensor_id}': {e}",
                level="ERROR"
            )

        log_system("[SensorManager] Target sensor thread setup complete.")

    def stop_all(self):
        """
        Stop all active BLE acquisition threads and clear the thread list.

        Calls BlueCoinThread.stop() on each thread, which sets the stop event
        and joins the thread (blocking until it exits cleanly).
        Safe to call even if initialize_sensors() was never called (empty list).
        """
        log_system("[SensorManager] Stopping active sensor threads...")

        for thread in self.threads:
            try:
                thread.stop()
            except Exception as e:
                log_system(
                    f"[SensorManager] Error stopping thread for device "
                    f"'{thread.device_id}': {e}",
                    level="ERROR"
                )

        self.threads.clear()
        log_system("[SensorManager] All sensor threads stopped.")

    def get_sensors_names(self) -> list:
        """
        Return the BLE advertisement names of all nodes found in the last scan.

        Used by main.py to check whether the expected device(s) are present
        before calling initialize_sensors(). Returns an empty list on error
        rather than raising, so the caller can handle missing sensors gracefully.
        """
        try:
            return [n.get_name() for n in self.bluecoins if n.get_name()]
        except Exception:
            return []
