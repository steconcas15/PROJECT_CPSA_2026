# actuation_policy.py
# Defines which actuators to activate based on the drowsiness events.
#
# Author: Francesco Urru
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT


from typing import List, Dict, Optional
import random
import time

from utils.audio_paths import AudioLibrary
from utils.config import get_policy_attempts

# Nuove costanti semantiche allineate al caso Drowsiness
TAG_SVEGLIO = 0
TAG_SLOW_DRIFT = 1
TAG_NOD = 2
TAG_SUDDEN_DROP = 3


class StereotipyActivationPolicy:
    """
    Event-driven loop per la prevenzione dei colpi di sonno alla guida:
      - Per tag 1, 2, 3: genera un'azione correttiva tramite gli attuatori disponibili.
      - Se lo stato persiste, varia la frequenza/tipo sullo stesso attuatore.
      - Se lo stato di sonnolenza cessa (ritorno a TAG_SVEGLIO), spegne le attuazioni attive.
    """

    def __init__(self, actuator_ids: List[str]):
        self.actuator_ids = list(actuator_ids or [])
        self.retries_per_actuator = max(1, int(get_policy_attempts()))
        self.rng = random.Random()

        # State across events
        self._current_tag: Optional[int] = None
        self._current_actuator: Optional[str] = None
        self._attempts_on_current: int = 0
        self._variation_idx: int = 0
        self._variation_idx_per_actuator: Dict[str, int] = {}

        # Variations - Stimoli visivi leggeri e forti
        self._led_variants_mild = [
            {"color": (255, 255, 255, 0), "intensity": 50, "speed": 100},
            {"color": (180, 220, 255, 0), "intensity": 60, "speed": 100},
            {"color": (200, 255, 200, 0), "intensity": 70, "speed": 100},
        ]
        self._led_variants_strong = [
            {"color": (255,   0,   0, 0), "intensity": 100, "speed": 100},
            {"color": (255, 128,   0, 0), "intensity": 100, "speed": 100},
            {"color": (255,   0, 255, 0), "intensity": 100, "speed": 100},
        ]
        # Variations - Stimoli vibrazione (metamotion) leggeri e forti
        self._meta_variants_mild = [
            {"duty": 40, "duration": 700},
            {"duty": 55, "duration": 800},
            {"duty": 70, "duration": 900},
        ]
        self._meta_variants_strong = [
            {"duty": 100, "duration": 900},
            {"duty": 100, "duration": 1100},
            {"duty": 100, "duration": 1300},
        ]
        # Variations - Audio (speaker) leggeri e forti
        self._spk_variants_mild = [
            {"file": getattr(AudioLibrary, "CUSTOM_MILD_1", getattr(AudioLibrary, "MISSING_FILE", ""))},
            {"file": getattr(AudioLibrary, "CUSTOM_MILD_2", getattr(AudioLibrary, "MISSING_FILE", ""))},
            {"file": getattr(AudioLibrary, "CUSTOM_MILD_3", getattr(AudioLibrary, "MISSING_FILE", ""))},
        ]
        self._spk_variants_strong = [
            {"file": getattr(AudioLibrary, "CUSTOM_STRONG_1", getattr(AudioLibrary, "MISSING_FILE", ""))},
            {"file": getattr(AudioLibrary, "CUSTOM_STRONG_2", getattr(AudioLibrary, "MISSING_FILE", ""))},
            {"file": getattr(AudioLibrary, "CUSTOM_STRONG_3", getattr(AudioLibrary, "MISSING_FILE", ""))},
        ]
        self._spk_cooldown_sec = 5
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: dict) -> Optional[Dict]:
        if not self.actuator_ids:
            return None

        tag = self._parse_tag(event.get("stereotipy_tag"))
        now = time.monotonic()

        # Ritorno allo stato SVEGLIO (0): Reset totale e spegnimento hardware
        off_action = None
        if tag == TAG_SVEGLIO or tag is None:
            if self._current_tag in (TAG_SLOW_DRIFT, TAG_NOD, TAG_SUDDEN_DROP) and self._current_actuator:
                if self._current_actuator.startswith("led_"):
                    off_action = {"actuator_id": self._current_actuator, "params": {"color": (0, 0, 0, 0), "intensity": 0, "speed": 100}}
            self._reset_state()
            return off_action

        # Primo evento positivo rilevato o cambio di tipologia/gravità
        if self._current_tag != tag or self._current_actuator is None:
            self._current_tag = tag
            self._attempts_on_current = 0

        # Raggiunto il limite di tentativi consecutivi sullo stesso attuatore: cambia dispositivo
        if self._attempts_on_current >= self.retries_per_actuator:
            self._current_actuator = self._pick_random(exclude=self._current_actuator)
            self._attempts_on_current = 0

        if self._current_actuator and self._current_actuator.startswith("speaker"):
            last = self._spk_last_fire_time.get(self._current_actuator, 0.0)
            if now - last < self._spk_cooldown_sec:
                return None

        # CONFIGURAZIONE INTENSITÀ STIMOLO:
        # Lo stimolo è impostato su "mild" (blando) SOLO per la deriva lenta (tag 1).
        # Per i colpi di sonno (tag 2) e crolli improvvisi della testa (tag 3) l'intensità è massima (strong).
        mild = (tag == TAG_SLOW_DRIFT)
        
        var_idx = self._get_variation_idx(self._current_actuator)
        result = self._params_for(self._current_actuator, mild=mild, variation_index=var_idx, now=now)
        if not result:
            # Fallback immediato su un altro attuatore se non configurato correttamente
            self._current_actuator = self._pick_random(exclude=self._current_actuator)
            self._attempts_on_current = 0
            var_idx = self._get_variation_idx(self._current_actuator)
            result = self._params_for(self._current_actuator, mild=mild, variation_index=var_idx, now=now)
            if not result:
                return None

        # Memorizzazione temporale dell'evento ed avanzamento variazioni
        self._spk_last_fire_time[self._current_actuator] = now
        self._attempts_on_current += 1
        self._bump_variation_idx(self._current_actuator)
        return result

    # ----- helpers -----
    def _parse_tag(self, raw) -> Optional[int]:
        try:
            return int(raw)
        except Exception:
            return None

    def _pick_random(self, exclude: Optional[str]) -> Optional[str]:
        candidates = self.actuator_ids
        if exclude and len(candidates) > 1:
            candidates = [a for a in candidates if a != exclude]
        return self.rng.choice(candidates) if candidates else None

    def _params_for(self, actuator_id: Optional[str], *, mild: bool, variation_index: int, now: float) -> Optional[Dict]:
        if not actuator_id:
            return None

        if actuator_id.startswith("led_"):
            variants = self._led_variants_mild if mild else self._led_variants_strong
            params = variants[variation_index % len(variants)]

        elif actuator_id.startswith("speaker_"):
            variants = self._spk_variants_mild if mild else self._spk_variants_strong

            try:
                params = variants[variation_index % len(variants)]
            except (KeyError, IndexError):
                params = {"file": getattr(AudioLibrary, "MISSING_FILE", "")}

        elif actuator_id.startswith("meta_"):
            variants = self._meta_variants_mild if mild else self._meta_variants_strong
            params = variants[variation_index % len(variants)]

        else:
            return None

        return {"actuator_id": actuator_id, "params": params}

    def _reset_state(self):
        self._current_tag = None
        self._current_actuator = None
        self._attempts_on_current = 0
        self._variation_idx = 0

    def _get_variation_idx(self, actuator_id: Optional[str]) -> int:
        if not actuator_id:
            return 0
        return self._variation_idx_per_actuator.get(actuator_id, 0)

    def _bump_variation_idx(self, actuator_id: Optional[str]):
        if not actuator_id:
            return
        self._variation_idx_per_actuator[actuator_id] = self._get_variation_idx(actuator_id) + 1