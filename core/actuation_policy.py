from typing import List, Dict, Optional
import time
import os
from utils.audio_paths import AudioLibrary
from utils.logger import log_system

"""
Policy engine for drowsiness alerts.

Validates the presence of the alert audio file at startup, monitors the video
pipeline for continuous "DROWSY" states, and handles a 5-second cooldown per
speaker to prevent overlapping or audio flooding. Returns the actuation payload.
"""

class DrowsyAlertPolicy:
    """
    Policy engine that determines when to trigger auditory alerts based on 
    drowsiness detection and per-actuator cooldown restrictions.
    """
    def __init__(self, actuator_ids: List[str]):
        # Filter and retain only the actuator IDs designated as speaker devices
        self.actuator_ids = [a for a in actuator_ids if a.startswith("speaker_")]
        
        # Retrieve the abstract audio track path from the central AudioLibrary.
        # Enforce string conversion to guarantee a human-readable/OS-compatible path format.
        raw_audio_path = str(AudioLibrary.DROWSINESS_ALERT)
        
        # Verification of the target audio file on disk
        if os.path.exists(raw_audio_path):
            self._audio_file = raw_audio_path
            log_system(f"[Policy] Audio caricato correttamente: {self._audio_file}")
        else:
            self._audio_file = ""
            log_system(f"[Policy] ERRORE: File audio non trovato al percorso: {raw_audio_path}", level="ERROR")

        # Cooldown parameters to prevent audio overlapping/flooding
        self._spk_cooldown_sec = 5
        # Maps each speaker's actuator_id to its last activation timestamp (monotonic time)
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: Dict, video_prediction: Optional[str]) -> Optional[Dict]:
        """
        Evaluates system context and video inference to issue an alert payload 
        if preconditions, states, and cooldown restrictions are satisfied.
        """

        # Guard clause: Abort instantly if the physical audio asset is missing/invalid
        if not self._audio_file:
            return None

        # Guard clause: Abort if no speaker hardware is available, 
        # or if the current computer vision prediction is not explicitly "DROWSY"
        if not self.actuator_ids or video_prediction != "DROWSY":
            return None

        now = time.monotonic()
        actuator_id = self.actuator_ids[0]

        # Extract the last time this specific speaker was triggered (defaults to 0.0)
        last_fire = self._spk_last_fire_time.get(actuator_id, 0.0)
        if now - last_fire < self._spk_cooldown_sec:
            return None

        # Enforce the per-speaker cooldown period
        self._spk_last_fire_time[actuator_id] = now

        # Dispatch execution command payload back to the EventDispatcher
        return {
            "actuator_id": actuator_id, 
            "params": {"file": self._audio_file}
        }
