# sensor_manager.py
# Manager for initializing and coordinating BlueCoin sensor threads

from blue_st_sdk.features.feature_accelerometer import FeatureAccelerometer
from blue_st_sdk.features.feature_gyroscope import FeatureGyroscope

from sensors.BLE.bluecoin import scan_bluecoin_devices, BlueCoinThread
from sensors.BLE.feature_listeners import AccelerometerFeatureListener, GyroscopeFeatureListener

from IMU_pipeline.data_stream.synchronizer import IMUSynchronizer
from IMU_pipeline.classifiers.drowsiness_classifier import DrowsinessClassifier

from utils.config import get_bluecoin_config
from utils.logger import log_system
from utils.lock import device_scan_lock, device_connection_lock


class SensorManager:
    """
    Manages initialization and coordination of a single BlueCoin sensor thread.
    Sinks streaming data into the Drowsiness Detection Python core.
    """
    def __init__(self):
        """Initializes the SensorManager and loads BlueCoin configuration."""
        self.threads = []
        self.config = get_bluecoin_config()
        self.bluecoins = []
        self.synchronizer = IMUSynchronizer()
        
        # Istanziazione del nuovo classificatore Python per sonnolenza
        self.classifier = DrowsinessClassifier()
        self.synchronizer.buffer.set_features_sink(self.classifier.recognize)
        log_system("[SensorManager] Initialized for single-device Drowsiness Detection pipeline")

    def scan_sensors(self):
        """Performs BLE scan for BlueCoin devices and stores results internally."""
        log_system("[SensorManager] Starting BLE scan for BlueCoin devices")
        with device_scan_lock:
            self.bluecoins = scan_bluecoin_devices(timeout=5)
        log_system(f"[SensorManager] Found {len(self.bluecoins)} BlueCoin device(s)")

        discovered = []
        for node in self.bluecoins:
            try:
                discovered.append(node.get_name())
            except Exception as e:
                log_system(f"[SensorManager] Error retrieving name for scanned node: {e}", level="WARNING")

    def initialize_sensors(self):
        """ Initializes single sensor thread for right BlueCoin device. """
        if not self.bluecoins:
            log_system("[SensorManager] No scanned bluecoins available. Run scan_sensors() first.", level="WARNING")
            return

        by_name = {}
        for node in self.bluecoins:
            try:
                name = node.get_name()
                if name:
                    by_name[name] = node
            except Exception as e:
                log_system(f"[SensorManager] Can't read node name: {e}", level="WARNING")

        expected = {c.get("id"): c.get("name") for c in self.config if c.get("id") and c.get("name")}
        right_name = expected.get("bc_right")

        if not right_name:
            log_system(f"[SensorManager] Config must include bc_right with a matching name.", level="ERROR")
            return

        if right_name not in by_name:
            log_system(f"[SensorManager] Missing expected right bluecoin device: '{right_name}'", level="ERROR")
            return

        # Setup del solo sensore destro
        sensor_id = "bc_right"
        expected_name = right_name
        node = by_name[expected_name]

        try:
            feat_acc = node.get_feature(FeatureAccelerometer)
            feat_gyr = node.get_feature(FeatureGyroscope)
        except Exception as e:
            log_system(f"[SensorManager] Error retrieving features for '{expected_name}': {e}", level="ERROR")
            return

        features, listeners = [], []
        if feat_acc:
            features.append(feat_acc)
            listeners.append(AccelerometerFeatureListener(device_id=sensor_id, synchronizer=self.synchronizer))
        else:
            log_system(f"[SensorManager] {expected_name} is missing Accelerometer", level="WARNING")

        if feat_gyr:
            features.append(feat_gyr)
            listeners.append(GyroscopeFeatureListener(device_id=sensor_id, synchronizer=self.synchronizer))
        else:
            log_system(f"[SensorManager] {expected_name} is missing Gyroscope", level="WARNING")

        if not features:
            log_system(f"[SensorManager] No target features available on node {expected_name}", level="WARNING")
            return

        try:
            with device_connection_lock:
                thread = BlueCoinThread(
                    node=node,
                    feature=features,
                    feature_listener=listeners,
                    device_id=sensor_id
                )
                thread.start()
                self.threads.append(thread)
            log_system(f"[SensorManager] Sensor initialized: {sensor_id} ({expected_name}) with {len(features)} active features")
        except Exception as e:
            log_system(f"[SensorManager] Error initializing thread for '{sensor_id}': {e}", level="ERROR")

        log_system("[SensorManager] Target sensor thread setup complete.")

    def stop_all(self):
        """Stops all active sensor threads and clears the list."""
        log_system("[SensorManager] Stopping active sensor threads...")
        for thread in self.threads:
            try:
                thread.stop()
            except Exception as e:
                log_system(f"[SensorManager] Error stopping thread for device '{thread.device_id}': {e}", level="ERROR")
        self.threads.clear()
        log_system("[SensorManager] All sensor threads stopped.")

    def get_sensors_names(self):
        try:
            return [n.get_name() for n in self.bluecoins if n.get_name()]
        except Exception:
            return []