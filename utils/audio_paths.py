# audio_paths.py
# Centralized symbolic mapping for audio files used by the speaker actuator
#
# Author: Francesco Urru
# GitHub: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

from pathlib import Path
from utils.logger import log_system

class AudioLibrary:
    _base_path = Path(__file__).resolve().parent.parent / "assets" / "audio"
    _files = {
        "DROWSINESS_ALERT": "beep_beep.mp3", #stereotipia_pericolosa_3
        "SPEAKER_CONNECT": "speaker_connected.mp3"
    }

    def __getattr__(self, name):
        if name in self._files:
            path = self._base_path / self._files[name]
            if not path.exists():
                log_system(f"[AudioLibrary] File non trovato: {path}", level="WARNING")
                return "" # Ritorna vuoto se il file manca davvero
            return str(path) # Ritorna la stringa del percorso
        
        log_system(f"[AudioLibrary] Chiave non trovata: {name}", level="ERROR")
        return "" 

AudioLibrary = AudioLibrary()
