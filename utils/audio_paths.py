# audio_paths.py
# Centralized symbolic mapping for audio files used by the speaker actuator
#
# Author: Francesco Urru
# GitHub: https://github.com/frarvo
# Repository: https://github.com/frarvo/CPSA_2026
# License: MIT

from pathlib import Path

class _AudioLibraryMock:
    """
    Versione disabilitata per la fase di test.
    Non esegue controlli sul disco e non genera warning.
    Restituisce un percorso vuoto/finto per evitare crash di sistema.
    """

    def __getattr__(self, name):
        """
        Qualsiasi richiesta (es. AudioLibrary.CUSTOM_MILD_1) 
        finisce qui e restituisce None in modo silenzioso.
        """
        # Restituiamo None. La maggior parte dei player audio ben scritti 
        # ignora il comando se il percorso del file è None.
        return None

# Singleton instance
AudioLibrary = _AudioLibraryMock()
