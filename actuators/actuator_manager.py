# actuator_manager.py
# Manages scanning, initialization and lifecycle of speaker actuator devices.
#
# Author: Fabio Piras and Sara Caddeo
# GitHub: https://github.com/steconcas15
# Repository: https://github.com/steconcas15/PROJECT_CPSA_2026
# License: MIT

import threading
from typing import Dict

from actuators.BT.speaker import scan_speaker_devices, SpeakerThread
from utils.logger import log_system
from utils.lock import device_scan_lock
from utils.config import get_speaker_config

class ActuatorManager:
    """
    Manages discovery, initialization, and control of a bluetooth speaker actuators.

    Provides a unified interface for triggering audio actions on connected speakers.
    """

    def __init__(self):
        self.actuators: Dict[str, threading.Thread] = {}
        self.speaker_addresses = []
        self.speaker_enable = {}
        log_system("[ActuatorManager] Initialized")

    def scan_actuators(self):
        """
        Scan wanted (from config.yaml) speaker devices and store their MAC addresses.
        """
        log_system("[ActuatorManager] Scanning for all actuator devices...")
        self.speaker_enable = (get_speaker_config() or {}).get("enable", True)

        if self.speaker_enable:
            with device_scan_lock:
                cfg = get_speaker_config() or {}
                mac = cfg.get("mac")
                if mac:
                    self.speaker_addresses = [mac]
                else:
                    devices = scan_speaker_devices(5)
                    self.speaker_addresses = devices if devices else []

    def initialize_actuators(self):
        """
        Initializes speaker threads using previously scanned MAC addresses.
        """
        log_system("[ActuatorManager] Initializing all actuator devices...")

        if self.speaker_enable:
            for mac in self.speaker_addresses:
                try:
                    actuator_id = f"speaker_{mac}"
                    thread = SpeakerThread(mac)
                    thread.start()
                    self.actuators[actuator_id] = thread
                    log_system(f"[ActuatorManager] Speaker initialized: {actuator_id}")
                except Exception as e:
                    log_system(f"[ActuatorManager] Speaker {mac} initialization failed: {e}", level="ERROR")


        log_system("[ActuatorManager] Initialization complete")

    def trigger(self, actuator_id: str, action_type: str, **kwargs):
        """
        Triggers an audio action on the specified speaker.

        Args:
            actuator_id (str): ID of the speaker (e.g., 'speaker_A1:B2:C3:D4:E5:F6')
            action_type (str): Type of action to perform (reserved for future use)
            **kwargs: Additional parameters for the action.
        """
        actuator = self.actuators.get(actuator_id)

        if not actuator:
            log_system(f"[ActuatorManager] Attempted to trigger unknown actuator: {actuator_id}", level="WARNING")
            return

        try:
            # Prevent actuation if device not connected
            if actuator_id.startswith("speaker_"):
                try:
                    if hasattr(actuator, "_is_connected") and not actuator._is_connected():
                        log_system(f"[ActuatorManager] Speaker not connected, skipping actuation", level="WARNING")
                        return
                except Exception:
                    if getattr(actuator, "connected", False) is False:
                        log_system(f"[ActuatorManager] Speaker not connected, skipping actuation", level="WARNING")
                        return

            actuator.execute(**kwargs)
            log_system(f"[ActuatorManager] Triggered action on {actuator_id}: {kwargs}")
        except Exception as e:
            log_system(f"[ActuatorManager] Error triggering actuator {actuator_id}: {e}", level="ERROR")

    def get_actuators_ids(self):
        """
        Returns all registered speaker IDs.
        :return: A list of speaker ID strings
        """
        return list(self.actuators.keys())

    def stop_all(self):
        """
        Stops all speaker threads and clears the registry.
        """
        log_system("[ActuatorManager] Stopping all actuator threads...")

        for actuator_id, thread in self.actuators.items():
            try:
                thread.stop()
                log_system(f"[ActuatorManager] Stopped actuator: {actuator_id}")
            except Exception as e:
                log_system(f"[ActuatorManager] Error stopping actuator {actuator_id}: {e}", level="ERROR")

        self.actuators.clear()
        log_system("[ActuatorManager] All actuator threads stopped.")
