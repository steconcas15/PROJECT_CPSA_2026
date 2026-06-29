from typing import List, Dict, Optional
import time
import os
from utils.audio_paths import AudioLibrary
from utils.logger import log_system

class DrowsyAlertPolicy:
    def __init__(self, actuator_ids: List[str]):
        self.actuator_ids = [a for a in actuator_ids if a.startswith("speaker_")]
        
        # Tentativo di recupero tramite AudioLibrary
        # Usiamo il metodo str() per essere sicuri di avere un path leggibile
        raw_audio_path = str(AudioLibrary.DROWSINESS_ALERT)
        
        # Verifica fisica dell'esistenza del file
        if os.path.exists(raw_audio_path):
            self._audio_file = raw_audio_path
            log_system(f"[Policy] Audio caricato correttamente: {self._audio_file}")
        else:
            self._audio_file = ""
            log_system(f"[Policy] ERRORE: File audio non trovato al percorso: {raw_audio_path}", level="ERROR")
        
        self._spk_cooldown_sec = 5
        self._spk_last_fire_time: Dict[str, float] = {}

    def handle(self, event: Dict, video_prediction: Optional[str]) -> Optional[Dict]:
        # Se non abbiamo un file valido, non attiviamo l'azione
        if not self._audio_file:
            return None

        if not self.actuator_ids or video_prediction != "DROWSY":
            return None

        now = time.monotonic()
        actuator_id = self.actuator_ids[0]

        last_fire = self._spk_last_fire_time.get(actuator_id, 0.0)
        if now - last_fire < self._spk_cooldown_sec:
            return None
        
        self._spk_last_fire_time[actuator_id] = now
        
        return {
            "actuator_id": actuator_id, 
            "params": {"file": self._audio_file}
        }
