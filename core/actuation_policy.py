# actuation_policy.py
#
# Policy engine responsible for managing drowsiness alerts. 
# It ensures the necessary audio files are available at startup, watches the video 
# pipeline for drowsiness indicators, and applies a brief cooldown period to each 
# speaker to prevent overlapping sounds before issuing the final execution command.

from typing import List, Dict, Optional
import time
import os
from utils.audio_paths import AudioLibrary
from utils.logger import log_system

class DrowsyAlertPolicy:
    """
    A policy engine designed to determine exactly when to sound auditory alerts, 
    balancing active drowsiness detections with strict cooldown rules for each speaker.
    """
    def __init__(self, actuator_ids: List[str]):
        # Isolates and retains only the actuators that actually function as speakers.
        self.actuator_ids = [a for a in actuator_ids if a.startswith("speaker_")]
        
        # Fetches the audio track's path from the central library, converting it 
        # to a standard string to ensure smooth operation across different operating systems.
        raw_audio_path = str(AudioLibrary.DROWSINESS_ALERT)
        
        # Verifies that the audio file actually exists on the disk before proceeding.
        if os.path.exists(raw_audio_path):
            self._audio_file = raw_audio_path
            log_system(f"[Policy] Audio file successfully loaded: {self._audio_file}")
        else:
            self._audio_file = ""
            log_system(f"[Policy] Error: Audio file not found at path: {raw_audio_path}", level="ERROR")

        # A 5-second cooldown prevents the system from spamming or playing overlapping alerts.
        self._spk_cooldown_sec = 5
        
        # This dictionary tracks the exact moment each speaker was last activated.
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: Dict, video_prediction: Optional[str]) -> Optional[Dict]:
        """
        Reviews the current system state and video predictions. If the user is classified 
        as drowsy and the speaker is not on cooldown, an appropriate alert payload is generated.
        """

        # Execution stops immediately if a valid audio file is not available.
        if not self._audio_file:
            return None

        # The alert is also canceled if no speakers are connected, 
        # or if the video feed does not currently classify the user as drowsy.
        if not self.actuator_ids or video_prediction != "DROWSY":
            return None

        now = time.monotonic()
        actuator_id = self.actuator_ids[0]

        # Retrieves the timestamp of the last trigger for this specific speaker, 
        # defaulting to a timestamp of 0.0 if it has never been used.
        last_fire = self._spk_last_fire_time.get(actuator_id, 0.0)
        
        # If the time elapsed since the last alert is shorter than the cooldown period, no action is taken.
        if now - last_fire < self._spk_cooldown_sec:
            return None

        # With the alert confirmed, the timestamp is updated to record this new activation.
        self._spk_last_fire_time[actuator_id] = now

        # Finally, the speaker ID and the audio file are bundled into a command for the event dispatcher.
        return {
            "actuator_id": actuator_id, 
            "params": {"file": self._audio_file}
        }
