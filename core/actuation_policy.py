# IMU_pipeline/policies/actuation_policy.py
# Event-driven activation policy for driver drowsiness interventions.
#
# Author: Francesco Urru (Adapted for Drowsiness Core)
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

from typing import List, Dict, Optional
import random
import time

from utils.audio_paths import AudioLibrary
from utils.config import get_policy_attempts

# Costanti semantiche native per il caso Drowsiness (Sonnolenza)
TAG_SVEGLIO = 0
TAG_SLOW_DRIFT = 1  # Deriva lenta della testa (Stadio iniziale)
TAG_NOD = 2         # Colpo di sonno / Oscillazione (Stadio intermedio)
TAG_SUDDEN_DROP = 3 # Crollo brusco della testa (Massimo pericolo)


class DrowsinessActivationPolicy:
    """
    Core logico per la mitigazione dei colpi di sonno alla guida:
      - Per tag 1 (Slow Drift): Genera stimoli blandi (MILD) per ridestare l'attenzione.
      - Per tag 2 e 3 (Nod / Sudden Drop): Genera stimoli intensi (STRONG) ad alto impatto.
      - Se lo stato persiste, varia ciclicamente la frequenza/tipo di stimolo sullo stesso hardware.
      - Se lo stato cessa (TAG_SVEGLIO), restituisce un comando di spegnimento totale (off_action).
    """

    def __init__(self, actuator_ids: List[str]):
        self.actuator_ids = list(actuator_ids or [])
        self.retries_per_actuator = max(1, int(get_policy_attempts()))
        self.rng = random.Random()

        # Gestione dello stato interno attraverso la sequenza di eventi
        self._current_tag: Optional[int] = None
        self._current_actuator: Optional[str] = None
        self._attempts_on_current: int = 0
        self._variation_idx_per_actuator: Dict[str, int] = {}

        # ------------------ CONFIGURAZIONE VARIANTI STIMOLI ------------------ #
        
        # 1. LED - Variazioni di colore, intensità e velocità (Mild vs Strong)
        self._led_variants_mild = [
            {"color": (255, 255, 255, 0), "intensity": 50, "speed": 100},  # Luce bianca soffusa
            {"color": (180, 220, 255, 0), "intensity": 60, "speed": 100},  # Luce azzurra intermittente
            {"color": (200, 255, 200, 0), "intensity": 70, "speed": 100},  # Luce verde di avviso
        ]
        self._led_variants_strong = [
            {"color": (255,   0,   0, 0), "intensity": 100, "speed": 200},  # Rosso flash rapido
            {"color": (255, 128,   0, 0), "intensity": 100, "speed": 200},  # Arancione strobo
            {"color": (255,   0, 255, 0), "intensity": 100, "speed": 250},  # Viola massima frequenza
        ]

        # 2. VIBRAZIONE (MetaMotion) - Duty cycle e durata dell'impulso (Mild vs Strong)
        self._meta_variants_mild = [
            {"duty": 40, "duration": 700},  # Vibrazione leggera pulsata
            {"duty": 55, "duration": 800},
            {"duty": 70, "duration": 900},
        ]
        self._meta_variants_strong = [
            {"duty": 100, "duration": 1000}, # Vibrazione continua al 100% della forza
            {"duty": 100, "duration": 1200},
            {"duty": 100, "duration": 1500},
        ]

        # 3. AUDIO (Speaker) - Tracce audio di avvertimento graduali (Mild vs Strong)
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
        
        self._spk_cooldown_sec = 4
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: dict) -> Optional[Dict]:
        """
        Elabora l'evento di sonnolenza in arrivo e mappa l'azione correttiva.
        Ritorna un dizionario contenente 'actuator_id' e i relativi 'params'.
        """
        if not self.actuator_ids:
            return None

        tag = self._parse_tag(event.get("stereotipy_tag"))
        now = time.monotonic()

        # RITORNO ALLO STATO SVEGLIO (0): Reset totale e spegnimento immediato dell'hardware attivo
        if tag == TAG_SVEGLIO or tag is None:
            off_action = None
            if self._current_tag in (TAG_SLOW_DRIFT, TAG_NOD, TAG_SUDDEN_DROP) and self._current_actuator:
                # Se l'attuatore corrente è un LED, invia esplicitamente il comando di spegnimento (RGB 0,0,0)
                if self._current_actuator.startswith("led_"):
                    off_action = {
                        "actuator_id": self._current_actuator, 
                        "params": {"color": (0, 0, 0, 0), "intensity": 0, "speed": 100}
                    }
            self._reset_state()
            return off_action

        # Primo evento positivo registrato o cambio improvviso di gravità/tipologia
        if self._current_tag != tag or self._current_actuator is None:
            self._current_tag = tag
            self._current_actuator = self._pick_random(exclude=None)
            self._attempts_on_current = 0

        # Raggiunto il limite massimo di tentativi consecutivi sullo stesso hardware: cambia dispositivo (escalation)
        if self._attempts_on_current >= self.retries_per_actuator:
            self._current_actuator = self._pick_random(exclude=self._current_actuator)
            self._attempts_on_current = 0

        # Protezione refrattaria (cooldown) specifica per gli allarmi acustici dello speaker
        if self._current_actuator and self._current_actuator.startswith("speaker_"):
            last_fire = self._spk_last_fire_time.get(self._current_actuator, 0.0)
            if now - last_fire < self._spk_cooldown_sec:
                return None

        # DETERMINAZIONE INTENSITÀ:
        # Lo stimolo è impostato su 'mild' (blando) SOLO per la deriva lenta (tag 1).
        # Per i colpi di sonno (tag 2) e crolli improvvisi (tag 3), l'intensità diventa massima ('strong').
        is_mild = (tag == TAG_SLOW_DRIFT)
        
        var_idx = self._get_variation_idx(self._current_actuator)
        result = self._params_for(self._current_actuator, mild=is_mild, variation_index=var_idx)
        
        # Se l'hardware selezionato fallisce la configurazione, esegui un fallback immediato su un altro attuatore
        if not result:
            self._current_actuator = self._pick_random(exclude=self._current_actuator)
            self._attempts_on_current = 0
            var_idx = self._get_variation_idx(self._current_actuator)
            result = self._params_for(self._current_actuator, mild=is_mild, variation_index=var_idx)
            if not result:
                return None

        # Aggiornamento dello storico temporale, avanzamento dei contatori e della variazione di stimolo
        self._spk_last_fire_time[self._current_actuator] = now
        self._attempts_on_current += 1
        self._bump_variation_idx(self._current_actuator)
        
        return result

    # ------------------------- METODI UTILITY INTERNAL ------------------------- #

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

    def _params_for(self, actuator_id: Optional[str], *, mild: bool, variation_index: int) -> Optional[Dict]:
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

    def _get_variation_idx(self, actuator_id: Optional[str]) -> int:
        if not actuator_id:
            return 0
        return self._variation_idx_per_actuator.get(actuator_id, 0)

    def _bump_variation_idx(self, actuator_id: Optional[str]):
        if not actuator_id:
            return
        self._variation_idx_per_actuator[actuator_id] = self._get_variation_idx(actuator_id) + 1
