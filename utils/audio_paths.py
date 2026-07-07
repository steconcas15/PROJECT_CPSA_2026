# audio_paths.py
#
# Author: Stefano Concas, Matteo Matta, Sara Cadedo, Fabio Piras
# Repository: https://github.com/steconcas15/PROJECT_CPSA_2026

from pathlib import Path
from utils.logger import log_system

class AudioLibrary:
    """
    Manages resolution and existence checks for system audio assets.
    """

    # Resolve absolute path to the local 'assets/audio' directory
    _base_path = Path(__file__).resolve().parent.parent / "assets" / "audio"

    # Internal map linking symbolic names to physical filenames
    _files = {
        "DROWSINESS_ALERT": "beep_beep.mp3",
        "SPEAKER_CONNECT": "speaker_connected.mp3"
    }

    def __getattr__(self, name):
        """
        Dynamically intercepts attribute access to resolve, validate, 
        and return the absolute string path of a requested audio track.
        """
        if name in self._files:
            path = self._base_path / self._files[name]

            # Verify file exists on disk before returning the path
            if not path.exists():
                log_system(f"[AudioLibrary] File non trovato: {path}", level="WARNING")
                return "" # Safe fallback if asset is missing
            return str(path) # Return validated path as a string
        
        log_system(f"[AudioLibrary] Chiave non trovata: {name}", level="ERROR")
        return "" 

AudioLibrary = AudioLibrary()
