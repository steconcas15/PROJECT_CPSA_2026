# actuation_policy.py
# Defines which actuators to activate based on the events.
#
# Author: Sara Caddeo
# Repository: https://github.com/steconcas15/PROJECT_CPSA_2026/core
# License: MIT


from typing import List, Dict, Optional
import random
import time

from utils.audio_paths import AudioLibrary
from utils.config import get_policy_attempts

TAG_AWAKE = 0
TAG_SUSPECT = 1
TAG_CONFIRMED = 2

class DrowsinessActivationPolicy:
    """
    Event-driven loop for Drowsiness:
      - Tag 0 (Awake) or 1 (Suspect): Reset state. No actuation.
      - Tag 2 (Confirmed): Emit actuation.
      - Sequence: Try MetaMotion (Haptic) for N attempts. 
      - If attempts exceed N, switch to Speaker (Audio) permanently until awake.
    """

    def __init__(self, actuator_ids: List[str]):
        self.actuator_ids = list(actuator_ids or [])
        self.max_vibration_attempts = max(1, int(get_policy_attempts()))

        self.meta_id = next((aid for aid in self.actuator_ids if aid.startswith("meta_")), None)
        self.speaker_id = next((aid for aid in self.actuator_ids if aid.startswith("speaker_")), None)
        
        # State across events
        self._current_tag: Optional[int] = None
        self._vibration_attempts: int = 0
        self._variation_idx_per_actuator: Dict[str, int] = {}


        # Variations
        self._meta_variants = [
            {"duty": 60, "duration": 800},
            {"duty": 80, "duration": 1200},
            {"duty": 100, "duration": 1500},
        ]
        self._spk_variants = [
            {"file": getattr(AudioLibrary, "CUSTOM_STRONG_1", getattr(AudioLibrary, "MISSING_FILE", ""))},
            {"file": getattr(AudioLibrary, "CUSTOM_STRONG_2", getattr(AudioLibrary, "MISSING_FILE", ""))},
        ]
        
        self._spk_cooldown_sec = 5
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: dict) -> Optional[Dict]:
        if not self.actuator_ids:
            return None

        tag = self._parse_tag(event.get("drowsiness_tag"))
        now = time.monotonic()

        # Back to non-stereotypy: reset.
        off_action = None
        if tag in (TAG_AWAKE, TAG_SUSPECT) or tag is None:
            self._reset_state()
            return None
            
        if tag == TAG_CONFIRMED:
            self._current_tag = tag
            actuator_to_use = None

            if self._vibration_attempts < self.max_vibration_attempts and self.meta_id:
                actuator_to_use = self.meta_id
                self._vibration_attempts += 1
                
            elif self.speaker_id:
                actuator_to_use = self.speaker_id
                last = self._spk_last_fire_time.get(actuator_to_use, 0.0)
                if now - last < self._spk_cooldown_sec:
                    return None
                self._spk_last_fire_time[actuator_to_use] = now

            if not actuator_to_use:
                return None

        # Build actuation
            var_idx = self._get_variation_idx(actuator_to_use)
            result = self._params_for(actuator_to_use, variation_index=var_idx)
            self._bump_variation_idx(actuator_to_use)
            
            return result

    # ----- helpers -----
    def _parse_tag(self, raw) -> Optional[int]:
        try:
            return int(raw)
        except Exception:
            return None

    def _params_for(self, actuator_id: str, variation_index: int) -> Optional[Dict]:
        if actuator_id.startswith("speaker_"):
            variants = self._spk_variants
            try:
                params = variants[variation_index % len(variants)]
            except (KeyError, IndexError):
                params = {"file": getattr(AudioLibrary, "MISSING_FILE", "")}
                
        elif actuator_id.startswith("meta_"):
            variants = self._meta_variants
            params = variants[variation_index % len(variants)]
            
        else:
            return None

        return {"actuator_id": actuator_id, "params": params}

    def _reset_state(self):
        self._current_tag = None
        self._current_actuator = None
        self._vibration_attempts = 0
        self._variation_idx_per_actuator.clear()

    def _get_variation_idx(self, actuator_id: Optional[str]) -> int:
        if not actuator_id:
            return 0
        return self._variation_idx_per_actuator.get(actuator_id, 0)

    def _bump_variation_idx(self, actuator_id: Optional[str]):
        if not actuator_id:
            return
        self._variation_idx_per_actuator[actuator_id] = self._get_variation_idx(actuator_id) + 1
